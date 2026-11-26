// fork: rms_norm (+ weight mul) that also writes the q8_1 copy of its output, for the matvecs that read it (any model).
// A decode layer runs rms_norm, mul and then quantize_q8_1 inside each matvec of the normed input; here one kernel writes both the f32 output
// and the q8_1 blocks into the matvec's reuse slot (q8reuse-fork.h), so the matvecs find it quantized already.
// The q8_1 blocks match quantize_q8_1 (quantize.cu): d = amax/127, q = round(x/d), ds = (d, d*sum q). Rows past ne0 up to the padding are zeros.
#include "common.cuh"

static __global__ void rms_norm_q8_kernel(const float * x, int64_t s_x, const float * w, float * dst, int64_t s_dst, block_q8_1 * q8,
        int ne0, int ne_pad, float eps) {
    const int row = blockIdx.x;
    const float * xr = x + row*s_x;
    float ss = 0.0f;
    for (int e = threadIdx.x; e < ne0; e += blockDim.x) { ss += xr[e]*xr[e]; }
    __shared__ float red[32];
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
    ss = warp_reduce_sum(ss);
    if (lane == 0) { red[warp] = ss; }
    __syncthreads();
    ss = warp_reduce_sum(lane < nw ? red[lane] : 0.0f);
    const float scale = rsqrtf(ss/ne0 + eps);
    float * dr = dst + row*s_dst;
    block_q8_1 * qr = q8 + (int64_t) row*(ne_pad/QK8_1);
    for (int e = threadIdx.x; e < ne_pad; e += blockDim.x) {   // blockDim a multiple of 32: a warp is one q8_1 block
        float y = 0.0f;
        if (e < ne0) {
            y = scale*xr[e];
            if (w) { y *= w[e]; }
            dr[e] = y;
        }
        const float amax = warp_reduce_max(fabsf(y));
        const float d = __fdiv_rn(amax, 127.0f);
        const int8_t q = amax == 0.0f ? 0 : roundf(__fdiv_rn(y, d));
        const int sum = warp_reduce_sum((int) q);
        qr[e/QK8_1].qs[e%QK8_1] = q;
        if (lane == 0) { qr[e/QK8_1].ds = make_half2(d, d*sum); }
    }
}

void ggml_cuda_rms_norm_q8_fork(const float * x, int64_t s_x, const float * w, float * dst, int64_t s_dst, void * q8,
        int ne0, int ne_pad, int rows, float eps, cudaStream_t stream) {
    rms_norm_q8_kernel<<<rows, 512, 0, stream>>>(x, s_x, w, dst, s_dst, (block_q8_1 *) q8, ne0, ne_pad, eps);
}

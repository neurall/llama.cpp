#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// fork: the compact slice of an expert is gathered in shared memory, at most one 4-byte entry per token. The buffer used to be sized for
// the whole pass, so an op over more tokens than a block's shared memory holds (smpbo/4: 25344 tokens with the 99 KiB of an RTX 3090)
// died in GGML_ASSERT(nbytes_shared <= smpbo) (a 31367-token dense split pass, whose expert card routes the experts of every card's
// tokens in one op). Now the tokens are walked in chunks of n_tokens_chunk (what the block holds, set by the launch from the device),
// each chunk's entries written out before the next chunk reuses the buffer. A chunk's output position needs the expert's start
// (nex_prev: the columns of all lower experts over ALL tokens), so with more than one chunk a first walk over the ids counts it.
// A pass that fits in one chunk (every pass before) runs the single walk as it did.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_tokens_chunk, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    const bool one_chunk = n_tokens <= n_tokens_chunk;

    int nex_prev = 0; // Number of columns for experts with a lower index.
    if (!one_chunk) {
        for (int i = threadIdx.x; i < n_tokens*n_expert_used; i += warp_size) {
            nex_prev += ids[(i / n_expert_used)*si1 + i % n_expert_used] < expert;
        }
        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    }

    int n_written = 0; // Columns of this expert written out by the chunks before.
    for (int it_start = 0; it_start < n_tokens; it_start += n_tokens_chunk) {
        const int it_stop = min(it_start + n_tokens_chunk, n_tokens);

        int nex_prev_chunk = 0; // Number of columns for experts with a lower index in this chunk (used when it is the only one).
        int it_compact     = 0; // Running index for the compact slice of this expert in this chunk.

        if constexpr (n_expert_used_template == 0) {
            // Generic implementation:
            for (int it = it_start; it < it_stop; ++it) {
                int iex_used = -1; // The index at which the expert is used, if any.
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    const int expert_used = ids[it*si1 + iex];
                    nex_prev_chunk += expert_used < expert;
                    if (expert_used == expert) {
                        iex_used = iex;
                    }
                }

                if (iex_used != -1) {
                    store[it_compact] = mm_ids_helper_store(it, iex_used);
                }

                if (warp_reduce_any<warp_size>(iex_used != -1)) {
                    it_compact++;
                }
            }
        } else {
            // Implementation optimized for specific numbers of experts used:
            // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
            constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
            static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
            for (int it0 = it_start; it0 < it_stop; it0 += warp_size/neu_padded) {
                const int it = it0 + threadIdx.x / neu_padded;

                const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
                const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < it_stop ?
                    ids[it*si1 + iex] : INT_MAX;
                const int iex_used = expert_used == expert ? iex : -1;
                nex_prev_chunk += expert_used < expert;

                // Whether the threads at this token position have used the expert:
                const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

                // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
                int it_compact_add_lower = 0;
#pragma unroll
                for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                    const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                    if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                        it_compact_add_lower += tmp;
                    }
                }

                if (iex_used != -1) {
                    store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
                }

                // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
                it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
            }
        }
        if (one_chunk) {
            nex_prev = warp_reduce_sum<warp_size>(nex_prev_chunk);
        }
        ggml_cuda_syncwarp();

        const int ic0 = nex_prev + n_written; // where this chunk's columns of the expert start
        for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
            const mm_ids_helper_store store_it = store[itc];
            const int it       = store_it.it();
            const int iex_used = store_it.iex_used();
            ids_dst[ic0 + itc] = it*n_expert_used + iex_used;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex_used] = ic0 + itc;
            } else {
                ids_src1[ic0 + itc] = it*sis1 + iex_used % nchannels_y;
            }
        }
        ggml_cuda_syncwarp(); // the next chunk reuses the store
        n_written += it_compact;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + n_written;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    // the whole pass when the block's shared memory holds it (one walk), else chunks of what it holds (see mm_ids_helper)
    const int n_tokens_chunk = (int) std::min<size_t>(n_tokens, smpbo/sizeof(mm_ids_helper_store));
    const size_t nbytes_shared = n_tokens_chunk*sizeof(mm_ids_helper_store);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_tokens_chunk, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

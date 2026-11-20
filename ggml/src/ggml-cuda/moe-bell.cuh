#include "common.cuh"

// fork: the decode bell (ggml/src/ggml-bell-fork.h): the card's two steps as kernels that read and write the layer's bell in mapped pinned host memory
void ggml_cuda_op_moe_bell(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_moe_bell_supported(const ggml_tensor * op);   // the op's bell is memory this process mapped for the cards (ggml_cuda_bell_alloc)

void * ggml_cuda_bell_alloc(int device, size_t size);   // host memory that card reads and writes by pointer; nullptr when the driver refuses
void   ggml_cuda_bell_free(void * ptr);

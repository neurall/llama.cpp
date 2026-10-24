// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

// rows of a gappy view into a compact tensor of the same shape: one transfer per row run of the source (fork; the layout of the token-major KV cache)
static bool ggml_backend_tensor_copy_rows(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (src->type != dst->type || !ggml_are_same_shape(src, dst) || src->nb[0] != dst->nb[0] || !ggml_is_contiguous(dst)) { return false; }
    const size_t row = ggml_row_size(src->type, src->ne[0]);   // bytes of a row (whole blocks for a quantized type)
    char * tmp = (char *) malloc(row);
    for (int64_t i3 = 0; i3 < src->ne[3]; i3++) { for (int64_t i2 = 0; i2 < src->ne[2]; i2++) { for (int64_t i1 = 0; i1 < src->ne[1]; i1++) {
        const size_t so = i3*src->nb[3] + i2*src->nb[2] + i1*src->nb[1], dof = i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1];
        ggml_backend_tensor_get(src, tmp, so, row);
        ggml_backend_tensor_set(dst, tmp, dof, row);
    } } }
    free(tmp);
    return true;
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!ggml_are_same_layout(src, dst) && ggml_backend_tensor_copy_rows(src, dst)) {
        return;
    }
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

static ggml_backend_moe_src_cb_t g_moe_src_cb = NULL;
static void *                    g_moe_src_ud = NULL;

void ggml_backend_set_moe_src_callback(ggml_backend_moe_src_cb_t cb, void * user_data) {
    g_moe_src_cb = cb;
    g_moe_src_ud = user_data;
}

static ggml_backend_moe_fill_cb_t g_moe_fill_cb = NULL;
static void *                     g_moe_fill_ud = NULL;

// fork: which experts of a staged range a card already holds (the expert cache's blocks on that card), for the prompt pass's staging fill (the prefetch upload
// in graph_compute_async): those are copied on the card instead of crossing the link again. Without it every pass uploaded every expert of every layer, also
// the ones the card's pool held (Qwen3.8-Flash-Next, 3k tokens: 49 GB per pass on the x16 card while its pool held about 40% of the experts). The query has no
// side effects (ggml_backend_moe_src counts a use per call); an expert whose block is still being uploaded counts as missing. Returns how many of [lo, hi) the
// card holds, data[id - lo] = the expert's bytes on the card or NULL, *buffer = the buffer they are in. Declared locally (no shared header, see g_xcopy_dst_ordered).
typedef int (*ggml_backend_moe_where_fn_t)(const struct ggml_tensor * weight, int32_t lo, int32_t hi, ggml_backend_dev_t dev, const void ** data,
                                           ggml_backend_buffer_t * buffer, void * user_data);
static ggml_backend_moe_where_fn_t g_moe_where_fn = NULL;
static void *                      g_moe_where_ud = NULL;
extern "C" {
    GGML_API void ggml_backend_set_moe_where_fn_fork(ggml_backend_moe_where_fn_t fn, void * user_data);
}
void ggml_backend_set_moe_where_fn_fork(ggml_backend_moe_where_fn_t fn, void * user_data) {
    g_moe_where_fn = fn;
    g_moe_where_ud = user_data;
}

// fork: bytes each card received from host memory for its expert matmuls (the staging fill's uploads, the range uploads of unstaged splits, the copy callback's
// runs), for the dense split's pass statistics: the balance turns them into the card's upload time (bytes / its measured link rate), and the run display shows
// stall = pass - compute - upload per card. Read and reset once per pass (ggml_backend_sched_upload_bytes_fork); any number of cards.
// All of a card's link traffic the scheduler issues, by kind (user 2026-10-24: the x4 card received 6-7 GB/s in parts of a pass that has no expert on it;
// what was it?): inputs from host memory (masks, positions, KV row indices: in_host), other cards' results (hand-offs, KV rows: in_card), its own results
// sent to other cards (out_card). Read and reset with the expert uploads, once per pass (ggml_backend_sched_link_bytes_fork).
struct sched_link_bytes { uint64_t up = 0, in_host = 0, in_card = 0, out_card = 0; };
// GGML_SCHED_PASS_STATS=1 (debugging and measurement runs only; run.py sets it unless -ns): the pass report's breakdown of every card's time (upload, wait, stall
// stamps) and link bytes. Off, none of it is recorded: no extra stream stamps, no counters; the card stamps the live balance needs stay (split start and end).
static bool ggml_backend_sched_pass_detail(void) {
    static const bool on = getenv("GGML_SCHED_PASS_STATS") && atoi(getenv("GGML_SCHED_PASS_STATS")) == 1;
    return on;
}
static std::mutex g_upb_mu;
static std::map<ggml_backend_dev_t, sched_link_bytes> g_upb;
static void ggml_backend_sched_upload_add(ggml_backend_dev_t dev, uint64_t n) {
    if (!ggml_backend_sched_pass_detail()) { return; }
    std::lock_guard<std::mutex> lk(g_upb_mu);
    g_upb[dev].up += n;
}
static void ggml_backend_sched_link_add(ggml_backend_dev_t dst, ggml_backend_dev_t src, uint64_t n) {   // src NULL: host memory
    if (!ggml_backend_sched_pass_detail()) { return; }
    std::lock_guard<std::mutex> lk(g_upb_mu);
    if (src == NULL) { g_upb[dst].in_host += n; return; }
    g_upb[dst].in_card += n;
    g_upb[src].out_card += n;
}
extern "C" {
    GGML_API uint64_t ggml_backend_sched_upload_bytes_fork(ggml_backend_dev_t dev);
    GGML_API void     ggml_backend_sched_upload_add_fork(ggml_backend_dev_t dev, uint64_t n);
    GGML_API void     ggml_backend_sched_link_bytes_fork(ggml_backend_dev_t dev, uint64_t * in_host, uint64_t * in_card, uint64_t * out_card);
    GGML_API void     ggml_backend_sched_link_add_fork(ggml_backend_dev_t dst, ggml_backend_dev_t src, uint64_t n);
}
void ggml_backend_sched_link_add_fork(ggml_backend_dev_t dst, ggml_backend_dev_t src, uint64_t n) { ggml_backend_sched_link_add(dst, src, n); }
uint64_t ggml_backend_sched_upload_bytes_fork(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(g_upb_mu);
    auto it = g_upb.find(dev);
    if (it == g_upb.end()) { return 0; }
    const uint64_t v = it->second.up;
    it->second.up = 0;
    return v;
}
void ggml_backend_sched_link_bytes_fork(ggml_backend_dev_t dev, uint64_t * in_host, uint64_t * in_card, uint64_t * out_card) {
    std::lock_guard<std::mutex> lk(g_upb_mu);
    sched_link_bytes & b = g_upb[dev];
    *in_host = b.in_host; *in_card = b.in_card; *out_card = b.out_card;
    b.in_host = b.in_card = b.out_card = 0;
}
void ggml_backend_sched_upload_add_fork(ggml_backend_dev_t dev, uint64_t n) { ggml_backend_sched_upload_add(dev, n); }

static ggml_backend_split_cb_t g_split_cb = NULL;
static void *                  g_split_ud = NULL;

// fork: a backend tells the scheduler that its copy between two of its devices (cpy_tensor_async) writes the destination tensor on the DESTINATION's own stream,
// i.e. in stream order behind everything queued on that card before. Such a copy needs no wait for the destination card at all (see ggml_backend_sched_copy_input:
// the stock code drains the destination on the host, the fork made the SOURCE stream wait for the destination's tail on the device, which parked the source card's
// compute behind the other card). Declared locally by the backend that sets it (no shared header: a header edit rebuilds every CUDA source).
typedef bool (*ggml_backend_xcopy_dst_ordered_t)(ggml_backend_t src, ggml_backend_t dst);
static ggml_backend_xcopy_dst_ordered_t g_xcopy_dst_ordered = NULL;
extern "C" {
    GGML_API void ggml_backend_sched_set_xcopy_dst_ordered(ggml_backend_xcopy_dst_ordered_t fn);
}
void ggml_backend_sched_set_xcopy_dst_ordered(ggml_backend_xcopy_dst_ordered_t fn) {
    g_xcopy_dst_ordered = fn;
}

// fork: device-time stamps of the expert chains of the multi-GPU prompt split, the measurement behind the share controller (llama_split_ctl in llama-graph.cpp).
// The host thread issues work several layers ahead of the cards, so nothing the host can time says when a card reached a point. A backend that can run a host
// function in stream order (CUDA: cudaLaunchHostFunc) registers g_stamp_fn; the function writes the wall clock into *dst when the card's stream gets there.
// Per layer and card, three stamps:
//     t_start  in front of the card's first expert matmul of the layer (its inputs have arrived)
//     t_end    behind its last one
//     t_out    behind its half of the copy that hands its partial sum to another card (0 for the card that keeps the result: the layer's owner)
// A helper is late by t_out(helper) - t_end(owner): positive, the owner waits for it; negative, the helper finished early and could hold more experts.
// One measurement at a time, of the last prompt graph issued; it is complete when every t_end has been written, and then becomes readable through
// ggml_backend_sched_chain_stats. A graph issued while the previous measurement is still running on the cards is not measured (nothing here ever waits).
struct ggml_sched_chain_stat {
    int            layer;
    ggml_backend_t backend;
    int            lo, hi;       // the expert ids this card holds in this layer: [lo, hi)
    int64_t        t_start, t_end, t_out;   // microseconds, ggml_time_us(); written from the driver's callback thread (aligned 64-bit stores)
    int64_t        n_tokens;
};
typedef void (*ggml_backend_stamp_fn_t)(ggml_backend_t backend, int64_t * dst);
static ggml_backend_stamp_fn_t g_stamp_fn = NULL;
#define GGML_SCHED_MAX_CHAIN_STATS 2048
static struct ggml_sched_chain_stat g_chain_buf[2][GGML_SCHED_MAX_CHAIN_STATS];
static int        g_chain_n[2]   = { 0, 0 };
static int        g_chain_live   = -1;   // index of the measurement still running on the cards, -1 none
static int        g_chain_done   = -1;   // index of the last complete one
static long       g_chain_gen    = 0;    // counts complete measurements
static std::mutex g_chain_mu;
extern "C" {
    GGML_API void ggml_backend_sched_set_stamp_fn(ggml_backend_stamp_fn_t fn);
    GGML_API int  ggml_backend_sched_chain_stats(struct ggml_sched_chain_stat * out, int cap, long * gen);
}
void ggml_backend_sched_set_stamp_fn(ggml_backend_stamp_fn_t fn) {
    g_stamp_fn = fn;
}
// the measurement that is still running becomes the readable one once all its chains have ended
static void ggml_backend_sched_chain_roll(void) {
    if (g_chain_live < 0) { return; }
    for (int i = 0; i < g_chain_n[g_chain_live]; i++) {
        if (g_chain_buf[g_chain_live][i].t_end == 0) { return; }
    }
    g_chain_done = g_chain_live;
    g_chain_live = -1;
    g_chain_gen++;
}
// Copies the last complete measurement (returns the number of entries, 0 when there is none) and its generation: a caller that remembers the generation
// sees each measurement once.
int ggml_backend_sched_chain_stats(struct ggml_sched_chain_stat * out, int cap, long * gen) {
    std::lock_guard<std::mutex> lk(g_chain_mu);
    ggml_backend_sched_chain_roll();
    if (gen) { *gen = g_chain_gen; }
    if (g_chain_done < 0 || out == NULL) { return 0; }
    const int n = std::min(cap, g_chain_n[g_chain_done]);
    for (int i = 0; i < n; i++) { out[i] = g_chain_buf[g_chain_done][i]; }
    return n;
}

// fork: the busy time of every card in a measured prompt pass, from the same stream-order stamps (g_stamp_fn), one in front of and one behind every split:
// the expert chains (a MUL_MAT_ID split with an expert range: the MoE part) and everything else (the dense part) apart. The front stamp is queued after the
// copies that bring the split's inputs from other cards, so a card's busy time leaves out the time it waits for another card; it keeps the waits for its
// own expert uploads (queued behind the stamp). The dense split balances its two ratios on these (llama-dsplit-fork.cpp): the share of the tokens on the
// cards' dense times, the share of the experts on their MoE times. Measured together with the expert chains (the same passes); GGML_SCHED_CARD_STATS=0
// turns it off (two host functions per split: GLM-5.3-Flash has about 2000 splits in a split pass).
// t0: front stamp (behind the split's input copies), tw: behind the wait for the split's staged experts (0 when nothing was staged for it), t1: behind the
// split. xin: the split reads a tensor another GPU computed (a hand-off); upin: one of its host weights is uploaded on the card's own compute stream (not
// staged). ut0 / ut1: the staging upload of this split's experts on the card's upload stream (the ring), from its start to its end (0 when not staged).
// nf / nl / ni / nn (debugging, GGML_SCHED_PASS_TRACE only): the names of the split's first and last op and of the first tensor it reads from another GPU, and its op
// count, copied here because the graph is gone when the timeline is printed. They show what each split computes and what the next card waits for.
struct ggml_sched_card_stamp { int backend_id; int moe; int xin; int upin; int64_t t0, tw, t1, ut0, ut1, up_host_us; int nn; char nf[40], nl[40], ni[40]; char ops[400]; };
// Every upload from host memory into a card's compute stream (user 2026-10-24: "if any uploads happen to a card we must see them as upload time"): the
// user inputs (the attention mask and the KV row indices of a range: hundreds of MB at long context), expert ranges uploaded on the stream itself, the copy
// callback's work. Stream stamps around each one of at least 1 MiB (an upload that blocks the host is timed on the host: up_host_us), each against the
// split it feeds: they run in the gap in front of that split, and that much of the gap is the card's upload time, not a wait.
struct ggml_sched_up_stamp { int entry; int64_t t0, t1; };
#define GGML_SCHED_MAX_UP_STAMPS 32768
static struct ggml_sched_up_stamp g_up_buf[2][GGML_SCHED_MAX_UP_STAMPS];
static int g_up_n[2] = { 0, 0 };
static int g_cs_cur_set = -1, g_cs_cur_entry = -1;   // the measurement and the card stamp entry of the split whose inputs are being copied now
// wait_dense_us / wait_moe_us: the card's idle time (2026-10-23, user: "collect each card's actual wait directly"): the gap between the end of its previous
// split and the front stamp of a dense / an expert split, summed. The front stamp is queued behind the copies that bring the split's inputs from the other
// cards, so the gap is the time the card sat idle for another card's result (plus that copy itself, plus any lag of the host issuing work); max_wait_us is
// the longest single gap. Balanced cards wait only for the hand-off copies; a card that waits much more than the others is the one with too few tokens.
// The same time by cause (user 2026-10-24: "stall is not stall", the x4 card's 13.9 s of 18 s were the expert card computing; every part measured, the
// parts add up to the pass): dense_us + moe_us are kernel time only (from tw, the wait for the staged experts left out); up_wait_us is the card waiting for
// its expert uploads (t0 -> tw of a staged split, and the gap before a split whose experts are uploaded on the compute stream itself); xwait_us the gaps
// before a split that reads another GPU's result (the card waits for that card, the hand-off copy included); host_us the other gaps (nothing queued: the
// host had not issued the work yet). link_us: the time the card's upload stream spent on staging uploads (ut0 -> ut1; it overlaps the compute).
struct ggml_sched_card_stat  { ggml_backend_t backend; int64_t dense_us, moe_us, first_us, last_us; int n_dense, n_moe; int64_t wait_dense_us, wait_moe_us, max_wait_us;
                               int64_t up_wait_us, xwait_us, host_us, link_us; };
#define GGML_SCHED_MAX_CARD_STAMPS 16384
#define GGML_SCHED_CARD_MAX_BE     16
static struct ggml_sched_card_stamp g_card_buf[2][GGML_SCHED_MAX_CARD_STAMPS];
static ggml_backend_t g_card_be[2][GGML_SCHED_CARD_MAX_BE];
static int  g_card_n[2]  = { 0, 0 };
static int  g_card_live  = -1;
static int  g_card_done  = -1;
static long g_card_gen   = 0;
extern "C" {
    GGML_API int ggml_backend_sched_card_stats(struct ggml_sched_card_stat * out, int cap, long * gen);
}
// one upload of at least 1 MiB into card b's stream, for the split being fed now (see ggml_sched_up_stamp): the front stamp now, the slot of the end stamp
// returned (NULL: not measured); ggml_backend_sched_up_end stamps the end behind the upload
static int64_t * ggml_backend_sched_up_begin(ggml_backend_t b, size_t nbytes) {
    const int s = g_cs_cur_set;
    if (s < 0 || g_cs_cur_entry < 0 || g_stamp_fn == NULL || nbytes < ((size_t) 1 << 20) || g_up_n[s] >= GGML_SCHED_MAX_UP_STAMPS) { return NULL; }
    struct ggml_sched_up_stamp & u = g_up_buf[s][g_up_n[s]++];
    u.entry = g_cs_cur_entry; u.t0 = 0; u.t1 = 0;
    g_stamp_fn(b, &u.t0);
    return &u.t1;
}
static void ggml_backend_sched_up_end(ggml_backend_t b, int64_t * t1) {
    if (t1 != NULL) { g_stamp_fn(b, t1); }
}
static void ggml_backend_sched_up_host(int64_t us) {   // an upload that blocked the host (the card's stream was drained before it)
    if (g_cs_cur_set >= 0 && g_cs_cur_entry >= 0) { g_card_buf[g_cs_cur_set][g_cs_cur_entry].up_host_us += us; }
}
static void ggml_backend_sched_card_roll(void) {   // under g_chain_mu
    if (g_card_live < 0) { return; }
    for (int i = 0; i < g_card_n[g_card_live]; i++) {
        if (g_card_buf[g_card_live][i].t1 == 0) { return; }
    }
    g_card_done = g_card_live;
    g_card_live = -1;
    g_card_gen++;
}
// The last complete measurement summed per card (returns the number of cards, 0 when there is none) and its generation.
int ggml_backend_sched_card_stats(struct ggml_sched_card_stat * out, int cap, long * gen) {
    std::lock_guard<std::mutex> lk(g_chain_mu);
    ggml_backend_sched_card_roll();
    if (gen) { *gen = g_card_gen; }
    if (g_card_done < 0 || out == NULL) { return 0; }
    struct ggml_sched_card_stat acc[GGML_SCHED_CARD_MAX_BE] = {};
    int64_t prev_t1[GGML_SCHED_CARD_MAX_BE] = {};   // the end of the card's previous split (stamps are in issue order, which is each stream's execution order)
    // the uploads in front of each split (stream-stamped ones by their entry, plus the host-timed ones on the entry itself), and where the first one started
    std::vector<int64_t> upe(g_card_n[g_card_done], 0), up0(g_card_n[g_card_done], 0);
    for (int i = 0; i < g_up_n[g_card_done]; i++) {
        const struct ggml_sched_up_stamp & u = g_up_buf[g_card_done][i];
        if (u.entry < 0 || u.entry >= g_card_n[g_card_done] || u.t0 <= 0 || u.t1 < u.t0) { continue; }
        upe[u.entry] += u.t1 - u.t0;
        up0[u.entry]  = up0[u.entry] == 0 ? u.t0 : std::min(up0[u.entry], u.t0);
    }
    for (int i = 0; i < g_card_n[g_card_done]; i++) {
        const struct ggml_sched_card_stamp & c = g_card_buf[g_card_done][i];
        if (c.backend_id < 0 || c.backend_id >= GGML_SCHED_CARD_MAX_BE || c.t0 == 0 || c.t1 < c.t0) { continue; }
        struct ggml_sched_card_stat & a = acc[c.backend_id];
        const int64_t up_in = upe[i] + c.up_host_us;   // uploads that ran in the gap in front of this split
        if (prev_t1[c.backend_id] > 0 && c.t0 > prev_t1[c.backend_id]) {
            const int64_t gap = c.t0 - prev_t1[c.backend_id];
            (c.moe ? a.wait_moe_us : a.wait_dense_us) += gap;
            a.max_wait_us = std::max(a.max_wait_us, gap);
            // the gap by cause: its uploads (measured), then a wait for another card's result if the split reads one, else nothing was queued (host)
            const int64_t u = std::min(up_in, gap);
            a.up_wait_us += u;
            a.link_us    += u;
            (c.xin ? a.xwait_us : a.host_us) += gap - u;
        } else if (prev_t1[c.backend_id] == 0 && up_in > 0) {
            // the card's first split of the pass: its uploads ran before it, from the first of them (part of the pass, not of the time before the card started)
            const int64_t start = up0[i] > 0 ? std::min(up0[i], c.t0) : c.t0 - up_in;
            const int64_t u = std::min(up_in, c.t0 - start);
            a.up_wait_us += u;
            a.link_us    += u;
            a.first_us = a.first_us == 0 ? start : std::min(a.first_us, start);
        }
        prev_t1[c.backend_id] = c.t1;
        const int64_t tw = c.tw >= c.t0 && c.tw <= c.t1 ? c.tw : c.t0;   // no stamp: nothing was staged for this split
        a.up_wait_us += tw - c.t0;
        if (c.ut1 > c.ut0 && c.ut0 > 0) { a.link_us += c.ut1 - c.ut0; }
        if (c.moe) { a.moe_us += c.t1 - tw; a.n_moe++; } else { a.dense_us += c.t1 - tw; a.n_dense++; }
        a.first_us = a.first_us == 0 ? c.t0 : std::min(a.first_us, c.t0);
        a.last_us  = std::max(a.last_us, c.t1);
    }
    // GGML_SCHED_PASS_TRACE=N (debugging, needs the pass statistics): the timeline of 36 consecutive splits from the middle of the first N big passes, one line per
    // split: which card ran it, dense or expert, whether it read another card's result, when it started and ended (ms from the pass's first split) and how long the card
    // had been idle before it. It shows whether the cards run side by side or in turn, which the per-card totals cannot (user 2026-10-24: the expert card waited
    // 29-36% of every pass for the helper whatever the shares).
    static int trace_left = -1;
    static long traced_gen = -1;
    if (trace_left < 0) { const char * e = getenv("GGML_SCHED_PASS_TRACE"); trace_left = e ? atoi(e) : 0; }
    int64_t tf = 0, tl = 0;   // the pass's first split start and last split end
    for (int i = 0; trace_left > 0 && i < g_card_n[g_card_done]; i++) {
        const struct ggml_sched_card_stamp & c = g_card_buf[g_card_done][i];
        if (c.t0 > 0 && c.t1 >= c.t0) { tf = tf == 0 ? c.t0 : std::min(tf, c.t0); tl = std::max(tl, c.t1); }
    }
    if (trace_left > 0 && g_card_gen != traced_gen && g_card_n[g_card_done] > 200 && tl - tf > 5000000) {   // big passes only (5 s or more): not the startup calibration's, nor a cold start's small ones
        traced_gen = g_card_gen;
        trace_left--;
        const int ne = g_card_n[g_card_done];
        int64_t pv[GGML_SCHED_CARD_MAX_BE] = {};
        const int a0 = ne / 2, a1 = std::min(ne, a0 + 36);
        GGML_LOG_WARN("split timeline: %d splits, entries %d-%d, ms from the first split\n", ne, a0, a1 - 1);
        for (int i = 0; i < a1; i++) {
            const struct ggml_sched_card_stamp & c = g_card_buf[g_card_done][i];
            if (c.backend_id < 0 || c.backend_id >= GGML_SCHED_CARD_MAX_BE || c.t0 == 0 || c.t1 < c.t0) { continue; }
            if (i >= a0) {
                ggml_backend_dev_t d = g_card_be[g_card_done][c.backend_id] ? ggml_backend_get_device(g_card_be[g_card_done][c.backend_id]) : NULL;
                GGML_LOG_WARN("  #%d %s %s%s %.1f-%.1f (%.1f ms) idle before %.1f | %d ops: %s .. %s | reads %s\n", i, d ? ggml_backend_dev_name(d) : "?", c.moe ? "experts" : "dense", c.xin ? " (reads another card)" : "",
                        (c.t0 - tf) / 1e3, (c.t1 - tf) / 1e3, (c.t1 - c.t0) / 1e3, pv[c.backend_id] > 0 && c.t0 > pv[c.backend_id] ? (c.t0 - pv[c.backend_id]) / 1e3 : 0.0, c.nn, c.nf, c.nl, c.ni);
                if (c.ops[0]) { GGML_LOG_WARN("        ops: %s\n", c.ops); }
            }
            pv[c.backend_id] = c.t1;
        }
    }
    int n = 0;
    for (int b = 0; b < GGML_SCHED_CARD_MAX_BE && n < cap; b++) {
        if (acc[b].n_dense + acc[b].n_moe == 0) { continue; }
        acc[b].backend = g_card_be[g_card_done][b];
        out[n++] = acc[b];
    }
    return n;
}

void ggml_backend_set_split_callback(ggml_backend_split_cb_t cb, void * user_data) {
    g_split_cb = cb;
    g_split_ud = user_data;
}

void ggml_backend_set_moe_fill_callback(ggml_backend_moe_fill_cb_t cb, void * user_data) {
    g_moe_fill_cb = cb;
    g_moe_fill_ud = user_data;
}

// the expert-copy code (sched_copy_experts in llama-context.cpp) asks the registered MoE cache through these
bool ggml_backend_moe_src(const struct ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev, const void ** data, ggml_backend_buffer_t * buffer) {
    return g_moe_src_cb && g_moe_src_cb(weight, expert, dev, data, buffer, g_moe_src_ud);
}

bool ggml_backend_moe_fill(const struct ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev, void ** data, ggml_backend_buffer_t * buffer) {
    return g_moe_fill_cb && g_moe_fill_cb(weight, expert, dev, data, buffer, g_moe_fill_ud);
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

static void (*g_nan_hook_fork)(void) = nullptr;
extern "C" { GGML_API void ggml_sched_set_nan_hook_fork(void (*fn)(void)) { g_nan_hook_fork = fn; } }   // the block form: GGML_API may expand to "extern", invalid after extern "C" on the same declaration

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// a view whose rows sit far apart (a layer's K or V inside the token-major KV record: one row per cell, the other layers' rows in between): its byte span is several times
// the bytes it holds. Copying such a view with its own strides moves, and allocates on the other card, the whole span.
static bool ggml_backend_tensor_is_gappy(const struct ggml_tensor * t) {
    if (ggml_is_contiguous(t) || t->nb[0] != ggml_type_size(t->type)) { return false; }
    return ggml_nbytes(t) > 2*ggml_row_size(t->type, t->ne[0])*ggml_nrows(t);
}

// creates a copy of the tensor with the same memory layout (a compact one, rows only, for a gappy view: the consumers read the copy through its own strides)
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    if (ggml_backend_tensor_is_gappy(tensor)) {
        return dup;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS
// fork: the most staging slots a ring can have (pf-slots up to 15 = five layers of three expert tensors). The limit is only the size of the tables: 4
// silently capped pf-slots=6. Two layers (6) is what the tests use; a deeper ring was tried because in the multi-GPU prompt split a card's slots are
// released only behind its hand-offs (nsys, Qwen3.8-Flash-Next 3k tokens, dense split: the x16's copy engine busy 30 of every 42 ms, waiting on slot
// releases) and LOST on x16 + x4: pf-slots=12 gave 1201-1279 prompt tokens/s against 1290-1394 with 6 (four passes each; the extra slots are pool blocks
// that otherwise hold experts the staging fill copies on the card). Left open for links where the upload is not what a pass waits for.
#define GGML_SCHED_MAX_PREFETCH_SLOTS 16
#endif

#ifndef GGML_SCHED_MAX_PF_RINGS
#define GGML_SCHED_MAX_PF_RINGS 4
#endif
// Prefetch state of ONE card: expert tensors of its splits are uploaded on `backend` (a second instance of the card, its own stream) into `n_slots` staging slots
// while the card computes the previous split. Each card that computes expert matmuls has its own ring: the multi-GPU prompt split gives every card a half of
// the experts, and a ring for one card only left the other card's half on the slow path (read ids, sync, copy, compute, strictly in turn).
struct sched_pf_ring {
    ggml_backend_dev_t      dev;
    int                     n_slots;                 // 0 = off for this card
    int                     cur;
    ggml_backend_t          backend;
    ggml_backend_event_t    ready[GGML_SCHED_MAX_PREFETCH_SLOTS];   // upload done
    ggml_backend_event_t    free [GGML_SCHED_MAX_PREFETCH_SLOTS];   // compute that read the slot done
    bool                    used [GGML_SCHED_MAX_PREFETCH_SLOTS];
    ggml_backend_buffer_t   buf  [GGML_SCHED_MAX_PREFETCH_SLOTS];   // slot buffers the scheduler owns
    size_t                  buf_size;
    ggml_backend_buffer_t   region_buf;              // host-provided region: slot i = region_off + i * region_slot
    size_t                  region_off;
    size_t                  region_cap;
    bool                    region_only;             // staging slots only from the region: no region that fits = no prefetch for this card
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    // fork: prefetch overlap (--prefetch-experts-slots): while split i computes, the full expert tensor of split i+1 (+lookahead) is uploaded into a
    // staging slot on a second stream of the same device, so a streamed prompt pass costs max(upload, compute) per layer instead of their sum.
    // Slots are buffers the scheduler allocates, or, when the host sets a region (ggml_backend_sched_set_prefetch_region), ranges inside a buffer it owns
    // (the expert cache lends them). Removed by the #29943 merge, ported back here onto the callback-copy loop.
    struct sched_pf_ring    prefetch_ring[GGML_SCHED_MAX_PF_RINGS];   // one staging ring per card that runs expert matmuls (its own stream, events, slots, lent region)
    int                     prefetch_n_rings;
    int                     prefetch_n_slots;        // configured number of slots per ring (0 = off); a ring drops to fewer when memory is short
    size_t                  prefetch_slot_bytes;     // largest staged expert tensor (padded), set while splits are scanned
    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;
    size_t max_splits;       // bound the context and id arrays were sized for (see ggml_backend_sched_new)

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ggml_backend_event_t tail_ev[GGML_SCHED_MAX_BACKENDS];   // fork: per backend, marks the tail of its stream for a device-side wait before a cross-card input copy (see ggml_backend_sched_copy_input)
    uint32_t xcopy_touched;                                  // fork: bit b = a copy between two GPUs was just queued with backend b as its source or destination (read and cleared by the split loop)
    uint32_t xcopy_src;                                      // fork: the same, source cards only (the stamp behind a card's hand-off, see ggml_sched_chain_stat)
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;

    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int graph_size;
    int prev_graph_size;

    int debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
// fork: a graph input as the scheduler must treat it: the tensor flagged GGML_TENSOR_FLAG_INPUT, or a view of one, through any chain of views. The dense split cuts
// the attention mask and the KV row indices per card with views (ds_rows in llama-graph.cpp); the flag lives only on the original, so the checks that read the
// flag alone took every slice for a computed tensor: its copy drained the destination card on the host (16 drains of ~240 ms in one 12k-token pass of
// Qwen3.8-Flash-Next, 3.85 s of 8.2 s, 2026-10-24) while the one-card graph, reading the flagged tensor itself, never did. Every check of the flag goes through here.
static bool ggml_backend_sched_is_user_input(const struct ggml_tensor * t) {
    for (; t != NULL; t = t->view_src) {
        if (t->flags & GGML_TENSOR_FLAG_INPUT) { return true; }
    }
    return false;
}
// fork: the user-input copies already filled in the graph being computed (cleared at the start of ggml_backend_sched_compute_splits; see
// ggml_backend_sched_copy_input). Per thread: a scheduler computes on the thread that called it.
static thread_local std::unordered_set<const struct ggml_tensor *> g_in_done;

static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (ggml_backend_sched_is_user_input(tensor)) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            // fork: inside an expert matmul split of the multi-GPU prompt split (first node a MUL_MAT_ID with an id range, op_params 8 and 9), a node that reads a tensor
            // computed on ANOTHER GPU and not yet copied to this card starts a new split. A split's inputs are copied before its first node runs, so without this the
            // matmul waited for an input only a later node reads: the layer owner's `down` matmul for the helper card's partial sum (read by the add behind it), a
            // helper's `down` matmul for the router weights (read by the mul behind it). nsys, OLMoE 3.4k tokens, x16 owner + x4 helper: the owner idle 8 ms per layer
            // in front of its own `down`. Any number of cards: every add of another card's partial sum becomes its own wait point on the owner. GGML_SCHED_XSPLIT=0 turns off.
            static const bool xsplit = !(getenv("GGML_SCHED_XSPLIT") && atoi(getenv("GGML_SCHED_XSPLIT")) == 0);
            if (xsplit && !need_new_split && node_backend_id == cur_backend_id && i > split->i_start &&
                    ggml_backend_dev_type(ggml_backend_get_device(sched->backends[cur_backend_id])) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                const struct ggml_tensor * first = graph->nodes[split->i_start];
                // The same cut in dense splits (Qwen3.8 60k, x4 + x16: 1698 -> 1766 t/s; GGML_SCHED_XSPLIT_DENSE=0 / --moe xsplit-dense=0 turns off). The pass trace of the Qwen3.8 prompt split showed the
                // helper's own norm and QKV matmul (12 ms, independent) held back until the expert card's 24 ms split had produced the conv tail that only the LAST
                // node of the helper's split reads.
                static const bool xdense = !(getenv("GGML_SCHED_XSPLIT_DENSE") && atoi(getenv("GGML_SCHED_XSPLIT_DENSE")) == 0);   // on; --moe xsplit-dense=0 turns off
                if (xdense || (first->op == GGML_OP_MUL_MAT_ID && first->op_params[9] > first->op_params[8])) {
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL || (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                            continue;
                        }
                        const int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != -1 && src_backend_id != cur_backend_id && tensor_id_copy(hash_id(src), cur_backend_id, 0) == NULL &&
                                ggml_backend_dev_type(ggml_backend_get_device(sched->backends[src_backend_id])) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                                !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                GGML_ASSERT((size_t) i_split < sched->max_splits && "more graph splits than the scheduler reserved room for (ggml_backend_sched_new caps them at 16384)");
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if (!ggml_backend_sched_is_user_input(leaf)) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const bool graph_size_changed = sched->prev_graph_size != sched->graph_size;

        if (sched->callback_realloc != NULL) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && !graph_size_changed;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// NVTX ranges (GGML_NVTX=1): the scheduler thread names what it is doing, Nsight Systems shows it with the CUDA calls it contains. Function pointers from the CUDA backend.
static void (*g_nvtx_push)(const char *) = NULL;
static void (*g_nvtx_pop)(void) = NULL;
static bool g_nvtx_on = false;
// The CUDA backend's range functions (GGML_NVTX=1, nsys timelines) are handed in by the library that can look the backend up (llama_context: ggml_backend_reg_by_name lives in
// the ggml library, not in ggml-base: a call from here is an unresolved symbol when ggml-base is linked as a DLL on Windows, and the Windows build failed on it).
GGML_API void ggml_backend_nvtx_set_fork(void (*push)(const char *), void (*pop)(void)) {
    g_nvtx_push = push;
    g_nvtx_pop  = pop;
    g_nvtx_on   = push && pop;
}
static void ggml_backend_nvtx_init() {}
#define NVTX_PUSH(...) do { if (g_nvtx_on) { char _b[200]; snprintf(_b, sizeof _b, __VA_ARGS__); g_nvtx_push(_b); } } while (0)
#define NVTX_POP() do { if (g_nvtx_on) { g_nvtx_pop(); } } while (0)
struct nvtx_scope { bool on; explicit nvtx_scope(const char * n) : on(g_nvtx_on) { if (on) { g_nvtx_push(n); } } ~nvtx_scope() { if (on) { g_nvtx_pop(); } } };   // a range that ends with the scope (also on early returns)

// Time the host thread spends in the blocking calls of the split loop, by call site (GGML_PF_STATS=1 prints them per big graph). Sites: 1 copy_input waits for the split's own
// card, 2 copy_input synchronous cross-card fallback, 3 the expert copy callback reading the router ids (llama_context::sched_copy_experts), 4 D2H fetches before a split,
// 5 the previous card before a split without inputs, 6 copy-ahead of the next split's inputs, 7 tracing, 8 end of an eval-callback split.
static std::atomic<long long> g_sync_us[10], g_sync_n[10];
void ggml_backend_sched_sync_stat_add(int site, long long us) {
    if (site >= 0 && site < 10) { g_sync_us[site].fetch_add(us); g_sync_n[site].fetch_add(1); }
}
#define SYNC_TIMED(site, call) do { NVTX_PUSH("WAIT site%d", site); const int64_t _t = ggml_time_us(); call; NVTX_POP(); ggml_backend_sched_sync_stat_add(site, ggml_time_us() - _t); } while (0)
// same, and with GGML_PF_STATS=2 the tensor and the backends of every wait over 1 ms are logged (which input makes the scheduler thread wait, and for whom)
#define SYNC_TIMED_NAMED(site, call, tname, sb, db) do { NVTX_PUSH("WAIT site%d %s %s->%s", site, tname, sb, db); const int64_t _t = ggml_time_us(); call; NVTX_POP(); const int64_t _d = ggml_time_us() - _t; ggml_backend_sched_sync_stat_add(site, _d); \
    static const bool _v = getenv("GGML_PF_STATS") && atoi(getenv("GGML_PF_STATS")) >= 2; \
    if (_v && _d > 1000) { GGML_LOG_WARN("sync-wait %lld us at site %d: input %s, from %s to %s abs %lld\n", (long long) _d, site, tname, sb, db, (long long) _t); } } while (0)

// fork: see the definitions below (the row padding behind the last expert of a copied expert tensor)
extern "C" {
    GGML_API void ggml_backend_tensor_zero_tail_fork(ggml_backend_t backend, struct ggml_tensor * tensor);
}
static size_t ggml_backend_sched_expert_pad(ggml_backend_sched_t sched, int backend_id, const struct ggml_tensor * in);

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input, std::vector<ggml_backend_t> & d2h_pending) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    // fork: a VIEW of a user input is a user input too. The dense split hands every card its slice of the attention mask and of the KV row indices (ds_rows in
    // llama-graph.cpp); a slice lives in the same host buffer, written before the graph, but has no flag of its own, so it took the generic path below, which for a
    // CPU source drains the destination card on the host: 16 drains of ~240 ms in one 12k-token split pass of Qwen3.8-Flash-Next (attn_inp_kq_mask, 3.85 s of an
    // 8.2 s pass; nsys 2026-10-24: the scheduler thread 5.2 s in cudaStreamSynchronize). The scheduler stood still, neither card got new work: the cards of the
    // dense split ran in turn, the x4 card idle 75-80% of every pass. The one-card graph reads the whole mask (flagged) and never took that path.
    const bool user_input = ggml_backend_sched_is_user_input(input);
    // fork: a user input goes to a card ONCE per graph. Every split that reads it gets it as an input, and all of them on one card share the same copy
    // (tensor_copy: one per input, card and copy index; it stays allocated until its last reader). The host data does not change while the graph runs, and the
    // first copy is ordered before the later readers on the card's stream, so the later copies only repeated the same bytes. Measured before (2026-10-24,
    // Qwen3.8-Flash-Next 60k prompt, dense split x16 + x4): the attention mask rows went to each card once per attention layer, 12.5 times per pass, 24 GB per
    // pass into the x4 card at 63k cells (its 3.8 s upload wait of a 22.5 s pass) and the same 24 GB into the x16. Grows with tokens x cells.
    if (user_input && !g_in_done.insert(input_cpy).second) {
        return;
    }
    if (user_input && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        ggml_backend_sched_link_add(ggml_backend_get_device(split_backend), NULL, ggml_nbytes(input));   // see sched_link_bytes
    }
    if (user_input) {
        // fork: a user input already sitting in host memory (positions, masks, output ids) is queued as an async upload on the destination card's own stream. Stream order puts
        // it after everything queued there before, which is what the drain below protects, and the host buffer is not rewritten before the next graph (the caller synchronizes
        // before setting new inputs). Measured (nsys + NVTX, OLMoE split over two cards): the drain before inp_pos held the scheduler thread 247 ms of a 396 ms pass.
        // GGML_SCHED_DRAIN=1 keeps the drain.
        static const bool drain_in = getenv("GGML_SCHED_DRAIN") && atoi(getenv("GGML_SCHED_DRAIN")) == 1;
        if (!drain_in && sched->events[split_backend_id][sched->cur_copy] == NULL && input->buffer && ggml_backend_buffer_is_host(input->buffer) &&
                split_backend->iface.set_tensor_async != NULL && ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
                ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            NVTX_PUSH("ASYNC user input %s -> %s", input->name, ggml_backend_name(split_backend));
            int64_t * ue = ggml_backend_sched_up_begin(split_backend, ggml_nbytes(input));   // the card's upload time (see ggml_sched_up_stamp)
            ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
            ggml_backend_sched_up_end(split_backend, ue);
            NVTX_POP();
            return;
        }
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            SYNC_TIMED_NAMED(1, ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        } else {
            SYNC_TIMED_NAMED(1, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        }
        NVTX_PUSH("TENSOR_COPY user input %s", input->name);
        const int64_t t_c = ggml_time_us();
        ggml_backend_tensor_copy(input, input_cpy);
        ggml_backend_sched_up_host(ggml_time_us() - t_c);
        NVTX_POP();
        return;
    }

    // fork: the experts of a range matmul of the multi-GPU prompt split that got no staging slot (no memory for the ring on this card). The stock path drains the
    // card on the HOST and then lets the copy callback read the router ids, again with a host wait: the scheduler thread stands still until this card is idle and
    // issues nothing for the other cards meanwhile (nsys, GLM 8k, x16 owner + x4 helper: the x16's own expert chain started 123 ms late in every layer). A big
    // batch uses every expert of the range, so the whole range is queued as one upload on the card's own compute stream: stream order puts it behind everything
    // that may still read that memory, and nothing waits on the host. The expert cache does not see these experts (as with the staging ring).
    // GGML_SCHED_RANGE_ASYNC=0 restores the callback.
    if (ggml_backend_sched_is_host_weight(input) && split->graph.n_nodes > 0 && split_backend->iface.set_tensor_async != NULL &&
            sched->events[split_backend_id][sched->cur_copy] == NULL &&
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        static const bool range_async = !(getenv("GGML_SCHED_RANGE_ASYNC") && atoi(getenv("GGML_SCHED_RANGE_ASYNC")) == 0);
        const struct ggml_tensor * n0 = split->graph.nodes[0];
        if (range_async && n0->op == GGML_OP_MUL_MAT_ID && n0->op_params[9] > n0->op_params[8] && n0->src[0] == input_cpy && input->ne[2] > 0 &&
                ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy)) {
            const int64_t lo = n0->op_params[8], hi = std::min<int64_t>(n0->op_params[9], input->ne[2]);
            if (hi > lo) {
                const size_t esz = input->nb[2];
                const size_t off = (size_t) lo*esz;
                const size_t pad = hi < input->ne[2] ? ggml_backend_sched_expert_pad(sched, split_backend_id, input) : 0;
                NVTX_PUSH("ASYNC expert range %s [%d,%d) -> %s", input->name, (int) lo, (int) hi, ggml_backend_name(split_backend));
                int64_t * ue = ggml_backend_sched_up_begin(split_backend, (size_t) (hi - lo)*esz + pad);
                ggml_backend_tensor_set_async(split_backend, input_cpy, (const char *) input->data + off, off, (size_t) (hi - lo)*esz + pad);
                ggml_backend_sched_upload_add(ggml_backend_get_device(split_backend), (uint64_t) (hi - lo)*esz);
                if (hi == input->ne[2]) {
                    ggml_backend_tensor_zero_tail_fork(split_backend, input_cpy);
                }
                ggml_backend_sched_up_end(split_backend, ue);
                NVTX_POP();
                return;
            }
        }
    }

    // wait for the split backend to finish using the input before overwriting it
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } else {
        // No copy events (one input copy: no pipeline parallelism, e.g. experts in host memory). The copy must not overwrite input_cpy before the destination card has finished
        // everything queued on it so far. The stock code drains the destination stream on the HOST here, which stops the scheduler thread until that card is idle: nothing else
        // gets queued meanwhile, on either card. Measured (nsys + NVTX, OLMoE 3.4k-token prompt, expert split over two cards): 271 ms of a 415 ms pass in these drains, for the
        // router ids, the normed hidden state, the router weights and the layer output crossing the cards.
        // Same ordering on the GPU instead: an event at the tail of the destination stream, and the SOURCE stream (where the async peer copy runs) waits for it. Only for an
        // input coming from another GPU backend with events; anything else (CPU inputs, backends without events) keeps the drain. GGML_SCHED_DRAIN=1 restores the drain everywhere.
        static const bool drain = getenv("GGML_SCHED_DRAIN") && atoi(getenv("GGML_SCHED_DRAIN")) == 1;
        bool waited = false;
        // Third form, the one used when the backend offers it (g_xcopy_dst_ordered): the copy runs in two halves through pinned host memory, the read on the source
        // stream and the WRITE on the destination's own stream. Stream order on the destination is then the whole protection, nobody waits for anybody here, and the
        // source card's compute is never parked behind the destination card. nsys, OLMoE 3.4k tokens, x16 owner + x4 helper, with the tail wait: the 110 KB router
        // weights for the x4's `down` sat in the x16's compute stream until the x4 had finished `gate` and `up` and one of its own expert uploads, about 11 ms per layer
        // with no kernel on the x16; the hidden state itself held the x16's stream for the 5.8 ms the x4's link needs to take it.
        if (!drain && input_backend != split_backend && split_backend->iface.cpy_tensor_async != NULL &&
                g_xcopy_dst_ordered != NULL && g_xcopy_dst_ordered(input_backend, split_backend)) {
            waited = true;
        }
        if (!waited && !drain && input_backend != split_backend && split_backend->iface.cpy_tensor_async != NULL &&
                ggml_backend_dev_type(ggml_backend_get_device(input_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            if (sched->tail_ev[split_backend_id] == NULL) {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(ggml_backend_get_device(split_backend), &props);
                if (props.caps.events) { sched->tail_ev[split_backend_id] = ggml_backend_event_new(ggml_backend_get_device(split_backend)); }
            }
            if (sched->tail_ev[split_backend_id] != NULL) {
                NVTX_PUSH("DEVICE-WAIT %s: %s waits for the tail of %s", input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
                ggml_backend_event_record(sched->tail_ev[split_backend_id], split_backend);
                ggml_backend_event_wait(input_backend, sched->tail_ev[split_backend_id]);
                NVTX_POP();
                waited = true;
            }
        }
        if (!waited) {
            SYNC_TIMED_NAMED(9, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        }
    }

    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input)) {
        int64_t * ue = ggml_backend_sched_up_begin(split_backend, ggml_nbytes(input));   // the experts the callback uploads (and copies from the card's cache)
        const bool done = sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data);
        ggml_backend_sched_up_end(split_backend, ue);
        if (done) {
            return;
        }
    }

    // GGML_SCHED_D2H_ASYNC=0 turns off: a CPU split's inputs from a GPU are fetched async (ordered after their producers on that
    // GPU's stream) and each source backend is synced once after all inputs, instead of a synchronous copy + sync per input tensor
    static const bool d2h_async = !(getenv("GGML_SCHED_D2H_ASYNC") && atoi(getenv("GGML_SCHED_D2H_ASYNC")) == 0);
    if (d2h_async && input_cpy->buffer && ggml_backend_buffer_is_host(input_cpy->buffer) &&
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            input_backend->iface.get_tensor_async && ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy)) {
        ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));
        if (std::find(d2h_pending.begin(), d2h_pending.end(), input_backend) == d2h_pending.end()) {
            d2h_pending.push_back(input_backend);
        }
        return;
    }

    {   // link traffic (see sched_link_bytes): a card's input from another card, or from host memory (a tensor computed on the CPU, a weight copied whole)
        ggml_backend_dev_t dd = ggml_backend_get_device(split_backend), sd = ggml_backend_get_device(input_backend);
        if (ggml_backend_dev_type(dd) == GGML_BACKEND_DEVICE_TYPE_GPU && dd != sd) {
            ggml_backend_sched_link_add(dd, ggml_backend_dev_type(sd) == GGML_BACKEND_DEVICE_TYPE_GPU ? sd : NULL, ggml_nbytes(input));
        }
    }
    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    // fork: a copy between two GPUs is a hand-off on both cards' links; the split loop releases the cards' staging slots behind it (see "release of a staging slot")
    if (input_backend != split_backend && ggml_backend_dev_type(ggml_backend_get_device(input_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        for (int b = 0; b < sched->n_backends; b++) {
            if (sched->backends[b] == input_backend || sched->backends[b] == split_backend) { sched->xcopy_touched |= 1u << b; }
            if (sched->backends[b] == input_backend) { sched->xcopy_src |= 1u << b; }
        }
    }
    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        SYNC_TIMED_NAMED(2, ggml_backend_synchronize(input_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            SYNC_TIMED_NAMED(2, ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        } else {
            SYNC_TIMED_NAMED(2, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        }
        NVTX_PUSH("TENSOR_COPY sync-copy %s %s->%s", input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend)); const int64_t _tc = ggml_time_us(); ggml_backend_tensor_copy(input, input_cpy); NVTX_POP(); ggml_backend_sched_sync_stat_add(2, ggml_time_us() - _tc);
        if (ggml_backend_dev_type(ggml_backend_get_device(input_backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) { ggml_backend_sched_up_host(ggml_time_us() - _tc); }   // from host memory: the card's upload time
    }
    // fork: a quantized tensor copied whole into this graph's memory (a dense weight one card reads from another card in a split graph) gets zeros behind it,
    // on the destination's stream before any matmul there reads past its last row (see ggml_backend_tensor_zero_tail_fork). The graph memory is not a COMPUTE
    // buffer any more (the compute alias places it in the expert pool, a WEIGHTS buffer), so the matmuls' own clear of a temporary tensor's padding is skipped.
    // Measured 2026-10-18 (GGML_SCHED_NAN_CHECK=1): Qwen3.8-Flash-Next, dense split after a 40k prompt, CUDA0's copy of blk.5.ffn_down_shexp [640, 2560]
    // IQ4_NL: NaN in the last element of every output row, then '!!!!'; the bytes behind the copy were what an expert upload or another tensor had left there.
    if (ggml_is_quantized(input_cpy->type) && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        ggml_backend_tensor_zero_tail_fork(split_backend, input_cpy);
    }
}



// the staging ring of card `dev`; created on first use with the configured slot count (NULL when the table of rings is full)
static struct sched_pf_ring * ggml_backend_sched_pf_ring_for(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    for (int r = 0; r < sched->prefetch_n_rings; r++) {
        if (sched->prefetch_ring[r].dev == dev) {
            return &sched->prefetch_ring[r];
        }
    }
    if (sched->prefetch_n_rings >= GGML_SCHED_MAX_PF_RINGS) {
        return NULL;
    }
    struct sched_pf_ring * R = &sched->prefetch_ring[sched->prefetch_n_rings++];
    R->dev     = dev;
    R->n_slots = sched->prefetch_n_slots;
    return R;
}

// ---- prefetch overlap helpers ----------------------------------------------------------------------------------------------------------------
// The split whose first node is a MUL_MAT_ID over a host-resident weight and a batch large enough that nearly every expert is used (ids >= 2 x
// experts): its full expert tensor is staged on a second stream while the previous split computes.
static bool ggml_backend_sched_prefetch_target(ggml_backend_sched_t sched, struct ggml_backend_sched_split * s, ggml_tensor ** input, ggml_tensor ** input_cpy, int * why = nullptr) {
    // why (GGML_PF_STATS): 1 split has no inputs, 2 its first node is not a MUL_MAT_ID, 3 its weight is not one of its inputs, 4 the weight is not host-resident, 5 few ids
    int dummy; if (!why) { why = &dummy; }
    if (s->graph.n_nodes == 0 || s->n_inputs == 0) {
        *why = 1;
        return false;
    }
    ggml_tensor * node = s->graph.nodes[0];
    if (node->op != GGML_OP_MUL_MAT_ID) {
        *why = 2;
        return false;
    }
    // The expert matmuls of the multi-GPU prompt split (each card an id range [lo, hi), op_params 8 and 9) are staged ahead like any other (per-card rings and lookahead).
    // GGML_PF_SPLIT=0 keeps them out: their copies are then the layer's own transfer, issued for both cards together by the copy-ahead before either chain is launched.
    // Measured (nsys, GLM 8k, 25% on the x4 card): without the staging the x16's upload of the layer (1 GB, 38 ms) ran in front of its compute, 229 gaps over 5 ms
    // with no kernel on either card, 8.4 s in all, the pass 20.75 s against 17.4 s with it.
    static const bool pf_split = !(getenv("GGML_PF_SPLIT") && atoi(getenv("GGML_PF_SPLIT")) == 0);
    if (!pf_split && node->op_params[9] > node->op_params[8]) {
        *why = 6;
        return false;
    }
    for (int i = 0; i < s->n_inputs; i++) {
        ggml_tensor * cand = s->inputs[i];
        if (ggml_backend_sched_is_user_input(cand)) {
            continue;
        }
        ggml_tensor * cand_cpy = tensor_copy(cand, s->backend_id, sched->cur_copy);
        if (node->src[0] != cand_cpy) {
            continue;
        }
        if (!ggml_backend_sched_is_host_weight(cand)) {
            *why = 4;
            return false;
        }
        const ggml_tensor * ids = node->src[2];
        if (ids->ne[0]*ids->ne[1] < 2*cand->ne[2]) {
            *why = 5;
            return false;
        }
        *input = cand;
        *input_cpy = cand_cpy;
        return true;
    }
    *why = 3;
    return false;
}

// fork: the row padding behind the last expert of a copied expert tensor.
// A quantized matmul on the device reads whole tiles of a row (512 values). For a row whose length is not a multiple of the tile it reads past the row's end:
// into the next row, for the last row of an expert into the next expert, and for the last row of the LAST expert past the end of the tensor, into the padding
// the buffer type adds to the allocation for exactly this. What is read there is multiplied by zeros, so only a NaN or an infinity matters (0 x NaN = NaN in
// the row's sum). A weights buffer zeroes that padding when the tensor is created. A copy made for one graph does not: a staging slot of the upload ring or
// the tensor's copy in the compute buffer holds whatever was there before.
// Measured: Qwen3.8-Flash-Next, `down` tensors [640, 2560, 512] IQ4_NL (640 is not a multiple of 512). While every pass staged the same ranges, the bytes
// behind a staged tensor were the tail of the same kind of tensor and decoded to finite numbers. After the cards' expert shares had changed between two
// passes (other slot sizes, other places) they decoded to NaN: 5 values of layer 11's output, the last row for the tokens routed to the last expert, and
// every later token through the recurrent layers (perplexity NaN in the second chunk, 5 runs of 6). The same hole is in every path that copies a tensor up
// to its end: each of them now writes zeros behind it (ggml_backend_tensor_zero_tail_fork).
void ggml_backend_tensor_zero_tail_fork(ggml_backend_t backend, struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer == NULL || tensor->data == NULL) { return; }
    const size_t nb   = ggml_nbytes(tensor);
    const size_t full = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(tensor->buffer), tensor);
    if (full <= nb) { return; }   // no padding: the rows are whole tiles, or the type is not quantized
    static const std::vector<uint8_t> zeros(16384, 0);
    const size_t n = std::min(full - nb, zeros.size());
    struct ggml_tensor t = {};
    t.type   = GGML_TYPE_I8;
    t.buffer = tensor->buffer;
    t.data   = (char *) tensor->data + nb;
    t.ne[0] = (int64_t) n; t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 1; t.nb[1] = t.nb[2] = t.nb[3] = n;
    ggml_backend_tensor_set_async(backend, &t, zeros.data(), 0, n);
}

// what the device may read behind the last expert of a range that does not end at the tensor's end: the same row padding, taken from the next expert
static size_t ggml_backend_sched_expert_pad(ggml_backend_sched_t sched, int backend_id, const struct ggml_tensor * in) {
    const size_t full = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(sched->backends[backend_id]), in);
    const size_t tail = full > ggml_nbytes(in) ? full - ggml_nbytes(in) : 0;
    return std::min<size_t>(in->nb[2], std::max<size_t>(512, tail));
}

// Bytes a staging slot needs for the expert matmul that is the first node of split `s` and reads the host tensor `in`: the whole tensor as the device reads it
// (padded rows), or, for a matmul that owns an id range [lo, hi) (multi-GPU prompt split), only that range with the same tail padding. Sized for the whole
// tensor, the slots of a card that holds 30% of the experts were 3.3 times too large: on GLM-5.3-Flash (789 MB per expert tensor, 6 slots) the card on the slow
// link needed 4.7 GB of lent blocks, got none, and ran without the upload ring (nsys: its uploads inside its compute stream, the other card idle 123 ms per
// layer). GGML_PF_COMPACT=0 restores whole-tensor slots.
static size_t ggml_backend_sched_prefetch_bytes(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * s, const struct ggml_tensor * in) {
    static const bool compact = !(getenv("GGML_PF_COMPACT") && atoi(getenv("GGML_PF_COMPACT")) == 0);
    const size_t full = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(sched->backends[s->backend_id]), in);
    const struct ggml_tensor * node = s->graph.nodes[0];
    if (compact && node->op_params[9] > node->op_params[8] && in->ne[2] > 0) {
        const int64_t lo = node->op_params[8], hi = std::min<int64_t>(node->op_params[9], in->ne[2]);
        const size_t tail = full > ggml_nbytes(in) ? full - ggml_nbytes(in) : 0;
        if (hi > lo) { return std::min(full, (size_t) (hi - lo)*in->nb[2] + std::max<size_t>(512, tail) + 512); }
    }
    return full;
}

// bytes of the largest staged expert tensor (as the device reads it: padded rows; a card's own id range in the multi-GPU prompt split)
static size_t ggml_backend_sched_prefetch_scan(ggml_backend_sched_t sched, int backend_id) {
    size_t max_size = 0;
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * s = &sched->splits[split_id];
        if (backend_id >= 0 && s->backend_id != backend_id) {
            continue;
        }
        ggml_tensor * in = NULL, * cpy = NULL;
        if (ggml_backend_sched_prefetch_target(sched, s, &in, &cpy)) {
            max_size = std::max(max_size, ggml_backend_sched_prefetch_bytes(sched, s, in));
        }
    }
    return max_size;
}

static void ggml_backend_sched_prefetch_disable(ggml_backend_sched_t sched, struct sched_pf_ring * R, ggml_backend_t split_backend) {
    GGML_UNUSED(sched);
    R->n_slots = 0;
    if (R->backend) {
        ggml_backend_synchronize(split_backend);
        ggml_backend_synchronize(R->backend);
    }
    for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
        if (R->buf[i]) {
            ggml_backend_buffer_free(R->buf[i]);
            R->buf[i] = NULL;
        }
        R->used[i] = false;
    }
}

// make n slots of at least `size` bytes available; fills slot_buf / slot_data of every slot. false: prefetch is off (no memory, no async events)
static bool ggml_backend_sched_prefetch_prepare(ggml_backend_sched_t sched, struct sched_pf_ring * R, ggml_backend_t split_backend, size_t size, int n,
        ggml_backend_buffer_t * slot_buf, uint8_t ** slot_data) {
    if (R->backend == NULL) {
        ggml_backend_dev_t dev = split_backend->device;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.events) {
            R->n_slots = 0;
            return false;
        }
        R->backend = ggml_backend_dev_init(dev, NULL);
        if (R->backend == NULL) {
            R->n_slots = 0;
            return false;
        }
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            R->ready[i] = ggml_backend_event_new(dev);
            R->free[i]  = ggml_backend_event_new(dev);
            if (R->ready[i] == NULL || R->free[i] == NULL) {
                R->n_slots = 0;
                return false;
            }
        }
    }
    {
        int bid = -1;
        for (int b = 0; b < sched->n_backends; b++) { if (sched->backends[b] == split_backend) { bid = b; } }
        size = GGML_PAD(std::max(size, ggml_backend_sched_prefetch_scan(sched, bid)), 256);   // this card's largest staged range
    }
    sched->prefetch_slot_bytes = std::max(sched->prefetch_slot_bytes, size);

    // slots inside a buffer the host owns (the expert cache lends the blocks): no allocation here. As many slots as the lent region holds, up to the
    // configured number: the region is what the pool could lend next to THIS graph, so a big micro-batch gets fewer slots instead of the micro-batch
    // being cut down to make room for all of them (MiMo-V2.6-Flash 3-bit, 4k prompt: 6 slots forced 512-token passes, 127 prompt tokens/s; one pass
    // of 4k without slots ran 544). At least 3: fewer cannot keep an upload in flight safely (fire_ahead keeps up to n_slots - 1 per card).
    if (R->region_buf != NULL && size > 0) {
        const int fit = (int) std::min<size_t>((size_t) sched->prefetch_n_slots, R->region_cap / size);
        if (fit >= 3) {
            if (R->n_slots != fit) {
                R->n_slots = fit;
                R->cur %= fit;
            }
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(R->region_buf) + R->region_off;
            for (int i = 0; i < fit; i++) {
                slot_buf[i]  = R->region_buf;
                slot_data[i] = base + (size_t) i * size;
            }
            return true;
        }
    }

    if (R->region_only) {
        return false;   // the host lends the staging memory and none is available for this graph: no prefetch, no allocation
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(split_backend);
    for (int i = 0; i < n; i++) {
        if (R->buf[i] == NULL || ggml_backend_buffer_get_size(R->buf[i]) < size) {
            ggml_backend_buffer_t nb = ggml_backend_buft_alloc_buffer(buft, size);
            if (nb == NULL) {
                if (i >= 2 && R->buf[0] != NULL && ggml_backend_buffer_get_size(R->buf[0]) >= size) {
                    R->n_slots = i;   // fewer slots, still overlapping
                    R->cur = 0;
                    break;
                }
                ggml_backend_sched_prefetch_disable(sched, R, split_backend);
                return false;
            }
            if (R->buf[i] != NULL) {
                ggml_backend_synchronize(split_backend);
                ggml_backend_synchronize(R->backend);
                ggml_backend_buffer_free(R->buf[i]);
            }
            R->buf[i]  = nb;
            R->used[i] = false;
        }
    }
    for (int i = 0; i < R->n_slots; i++) {
        slot_buf[i]  = R->buf[i];
        slot_data[i] = (uint8_t *) ggml_backend_buffer_get_base(R->buf[i]);
    }
    return R->n_slots > 0;
}

// GGML_SCHED_NAN_CHECK=1 (debug, slow): after every split of a big graph its card is drained and the float tensors the split leaves behind are read back: those
// a later split reads (as a node input or as a copied input) and the split's last node. The others are skipped: their memory may already hold a later tensor of
// the same split, of another type (read as floats, an id tensor looks like NaNs). The first tensor that holds a NaN is reported once per process:
// graph number, split, card, node, operation, name, shape, how many values are bad and where the first one sits, and for each of its inputs whether that input
// holds bad values now. The drains change the timing: a fault that comes from two streams racing may not show.
static void ggml_backend_sched_nan_check(ggml_backend_sched_t sched, int split_id, long graph_no) {
    static bool reported = false;
    if (reported) { return; }
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    ggml_backend_t be = sched->backends[split->backend_id];
    ggml_backend_synchronize(be);
    static std::unordered_map<const struct ggml_tensor *, int> split_of;   // node -> its split
    static std::unordered_map<const struct ggml_tensor *, int> read_by;    // tensor -> the last split that reads it
    if (split_id == 0) {
        split_of.clear(); read_by.clear();
        for (int si = 0; si < sched->n_splits; si++) {
            const struct ggml_backend_sched_split * sp = &sched->splits[si];
            for (int j = 0; j < sp->graph.n_nodes; j++) { split_of[sp->graph.nodes[j]] = si; }
        }
        for (int si = 0; si < sched->n_splits; si++) {
            const struct ggml_backend_sched_split * sp = &sched->splits[si];
            for (int i = 0; i < sp->n_inputs; i++) {
                for (const struct ggml_tensor * t = sp->inputs[i]; t != NULL; t = t->view_src) { read_by[t] = si; }
            }
            for (int j = 0; j < sp->graph.n_nodes; j++) {
                for (int k = 0; k < GGML_MAX_SRC; k++) {
                    for (const struct ggml_tensor * t = sp->graph.nodes[j]->src[k]; t != NULL; t = t->view_src) { read_by[t] = si; }
                }
            }
        }
    }
    static std::vector<uint8_t> buf;
    const auto count_bad = [&](const struct ggml_tensor * t, int64_t * first) -> int64_t {
        if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || t->buffer == NULL || t->data == NULL || !ggml_is_contiguous(t)) { return -1; }
        const size_t nb = ggml_nbytes(t);
        if (nb == 0 || nb > ((size_t) 3 << 30)) { return -1; }
        buf.resize(nb);
        ggml_backend_tensor_get(t, buf.data(), 0, nb);
        int64_t bad = 0;
        *first = -1;
        if (t->type == GGML_TYPE_F32) {
            const uint32_t * v = (const uint32_t *) buf.data();
            for (size_t i = 0; i < nb / 4; i++) { if ((v[i] & 0x7f800000u) == 0x7f800000u && (v[i] & 0x007fffffu) != 0) { if (bad++ == 0) { *first = (int64_t) i; } } }
        } else {
            const uint16_t * v = (const uint16_t *) buf.data();
            for (size_t i = 0; i < nb / 2; i++) { if ((v[i] & 0x7c00u) == 0x7c00u && (v[i] & 0x03ffu) != 0) { if (bad++ == 0) { *first = (int64_t) i; } } }
        }
        return bad;
    };
    for (int j = 0; j < split->graph.n_nodes; j++) {
        const struct ggml_tensor * t = split->graph.nodes[j];
        if (ggml_is_view_op(t->op) || t->view_src != NULL) { continue; }
        {
            auto rb = read_by.find(t);
            if (j != split->graph.n_nodes - 1 && (rb == read_by.end() || rb->second <= split_id)) { continue; }   // nothing after this split reads it
        }
        int64_t first = -1;
        const int64_t bad = count_bad(t, &first);
        if (bad <= 0) { continue; }
        reported = true;
        const int64_t n = ggml_nelements(t);
        {   // --moe poison=3 leaves the name of the filling code (4 letters) and the serial of its fill call in a few words: the ones inside the bad tensor say who filled its bytes
            std::string who;
            for (size_t i = 0; i + 8 <= buf.size(); i += 4) {
                bool up = true; for (int k = 0; k < 4; ++k) { up = up && buf[i + k] >= 'A' && buf[i + k] <= 'Z'; }
                if (!up) { continue; }
                uint32_t sv; memcpy(&sv, &buf[i + 4], 4);
                const std::string tg((const char *) &buf[i], 4);
                if (who.find(tg) == std::string::npos) { who += " " + tg + "@" + std::to_string(sv); }
            }
            if (!who.empty()) { GGML_LOG_WARN("nan-check: poison writers inside %s (tag@fill serial):%s\n", t->name, who.c_str()); }
            std::string addrs; int na = 0;   // poison=4: 8-byte call-site addresses of the filling code (high word 0x5000..0x7fff); resolve with the ledger's "fill" lines (ra, anchor)
            for (size_t i = 0; i + 8 <= buf.size() && na < 6; i += 4) {
                uint32_t a0, a1; memcpy(&a0, &buf[i], 4); memcpy(&a1, &buf[i + 4], 4);
                if (a1 >= 0x5000 && a1 <= 0x7fff) { char b[40]; snprintf(b, sizeof b, " 0x%llx", (unsigned long long) (((uint64_t) a1 << 32) | a0)); if (addrs.find(b) == std::string::npos) { addrs += b; na++; } }
            }
            if (!addrs.empty()) { GGML_LOG_WARN("nan-check: poison call sites inside %s:%s\n", t->name, addrs.c_str()); }
        }
        GGML_LOG_WARN("nan-check: graph %ld, split %d of %d on %s, node %d of %d: %s %s [%lld,%lld,%lld,%lld] %s holds %lld bad values of %lld, the first at element %lld (row %lld of dim 1, %lld of dim 2)\n",
                graph_no, split_id, sched->n_splits, ggml_backend_name(be), j, split->graph.n_nodes, ggml_op_name(t->op), t->name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type), (long long) bad, (long long) n,
                (long long) first, (long long) (t->ne[0] > 0 ? (first / t->ne[0]) % std::max<int64_t>(1, t->ne[1]) : 0),
                (long long) (t->ne[0] * t->ne[1] > 0 ? first / (t->ne[0] * t->ne[1]) : 0));
        GGML_LOG_WARN("nan-check: op_params %d %d .. range [%d,%d); first node of the split: %s %s\n", (int) t->op_params[0], (int) t->op_params[1],
                (int) t->op_params[8], (int) t->op_params[9], ggml_op_name(split->graph.nodes[0]->op), split->graph.nodes[0]->name);
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const struct ggml_tensor * src = t->src[k];
            if (src == NULL) { continue; }
            int64_t f2 = -1;
            const int64_t b2 = count_bad(src->view_src && !ggml_is_contiguous(src) ? src->view_src : src, &f2);
            GGML_LOG_WARN("nan-check:   input %d: %s %s [%lld,%lld,%lld,%lld] %s, buffer %s: %s\n", k, ggml_op_name(src->op), src->name,
                    (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3], ggml_type_name(src->type),
                    src->buffer ? ggml_backend_buffer_name(src->buffer) : "-",
                    b2 < 0 ? "not checked (not a contiguous float tensor)" : b2 == 0 ? "no bad values" : "holds bad values");
            if (src->buffer) { GGML_LOG_WARN("nan-check:     input %d sits in a buffer of %zu bytes at offset %zu (data %p)\n", k, ggml_backend_buffer_get_size(src->buffer), (size_t) ((const char *) src->data - (const char *) ggml_backend_buffer_get_base(src->buffer)), (const void *) src->data); }
        }
        if (g_nan_hook_fork) { g_nan_hook_fork(); }   // --moe poison: dump the pool while the bad bytes are still there (after the report above: the hook exits the process)
        return;
    }
}

// GGML_SCHED_NAN_CHECK=2 (debug, very slow): every node of a big graph is run on its own and its output is read back before any later node can reuse memory.
// The first float output that holds a NaN is reported (an infinity is a legitimate value: masked scores) with the state of each of its inputs at that moment: bad values in a float input, the
// smallest and largest value of an integer input (an expert id outside the tensor reads memory that is not an expert). One report per process.
static bool ggml_backend_sched_nan_check_node(ggml_backend_sched_t sched, int split_id, int j, long graph_no) {
    static bool reported = false;
    if (reported) { return true; }
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    ggml_backend_t be = sched->backends[split->backend_id];
    const struct ggml_tensor * t = split->graph.nodes[j];
    ggml_backend_synchronize(be);
    if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || t->buffer == NULL || t->data == NULL || !ggml_is_contiguous(t) || ggml_is_view_op(t->op)) { return false; }
    static std::vector<uint8_t> buf;
    const auto read = [&](const struct ggml_tensor * x) -> bool {
        const size_t nb = ggml_nbytes(x);
        if (x->buffer == NULL || x->data == NULL || !ggml_is_contiguous(x) || nb == 0 || nb > ((size_t) 3 << 30)) { return false; }
        buf.resize(nb);
        ggml_backend_tensor_get(x, buf.data(), 0, nb);
        return true;
    };
    const auto count_bad = [&](const struct ggml_tensor * x, int64_t * first) -> int64_t {
        const size_t nb = ggml_nbytes(x);
        int64_t bad = 0; *first = -1;
        if (x->type == GGML_TYPE_F32) {
            const uint32_t * v = (const uint32_t *) buf.data();
            for (size_t i = 0; i < nb / 4; i++) { if ((v[i] & 0x7f800000u) == 0x7f800000u && (v[i] & 0x007fffffu) != 0) { if (bad++ == 0) { *first = (int64_t) i; } } }
        } else {
            const uint16_t * v = (const uint16_t *) buf.data();
            for (size_t i = 0; i < nb / 2; i++) { if ((v[i] & 0x7c00u) == 0x7c00u && (v[i] & 0x03ffu) != 0) { if (bad++ == 0) { *first = (int64_t) i; } } }
        }
        return bad;
    };
    if (!read(t)) { return false; }
    int64_t first = -1;
    const int64_t bad = count_bad(t, &first);
    if (bad <= 0) { return false; }
    reported = true;
    const int64_t per2 = t->ne[0] * t->ne[1];
    GGML_LOG_WARN("nan-node: graph %ld, split %d of %d on %s, node %d of %d: %s %s [%lld,%lld,%lld,%lld] %s holds %lld bad values of %lld, the first at element %lld (index %lld of dim 2, %lld of dim 1)\n",
            graph_no, split_id, sched->n_splits, ggml_backend_name(be), j, split->graph.n_nodes, ggml_op_name(t->op), t->name,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type), (long long) bad, (long long) ggml_nelements(t),
            (long long) first, (long long) (per2 > 0 ? first / per2 : 0), (long long) (t->ne[0] > 0 ? (first / t->ne[0]) % (t->ne[1] > 0 ? t->ne[1] : 1) : 0));
    GGML_LOG_WARN("nan-node:   op_params %d %d, expert range [%d,%d); first node of the split: %s %s\n", (int) t->op_params[0], (int) t->op_params[1],
            (int) t->op_params[8], (int) t->op_params[9], ggml_op_name(split->graph.nodes[0]->op), split->graph.nodes[0]->name);
    for (int k = 0; k < GGML_MAX_SRC; k++) {
        const struct ggml_tensor * src = t->src[k];
        if (src == NULL) { continue; }
        char what[160] = "not read (quantized, not contiguous or too big)";
        if (src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16) {
            if (read(src)) { int64_t f2 = -1; const int64_t b2 = count_bad(src, &f2); snprintf(what, sizeof what, "%lld bad values (first at element %lld)", (long long) b2, (long long) f2); }
        } else if (src->type == GGML_TYPE_I32) {
            if (read(src)) {
                const int32_t * v = (const int32_t *) buf.data(); const size_t n = ggml_nbytes(src) / 4;
                int32_t mn = n ? v[0] : 0, mx = n ? v[0] : 0; size_t n_neg = 0;
                for (size_t i = 0; i < n; i++) { mn = v[i] < mn ? v[i] : mn; mx = v[i] > mx ? v[i] : mx; n_neg += v[i] < 0; }
                snprintf(what, sizeof what, "values %d .. %d, %zu of %zu negative", mn, mx, n_neg, n);
            }
        }
        GGML_LOG_WARN("nan-node:   input %d: %s %s [%lld,%lld,%lld,%lld] %s at %p, buffer %s: %s\n", k, ggml_op_name(src->op), src->name,
                (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3], ggml_type_name(src->type), (void *) src->data,
                src->buffer ? ggml_backend_buffer_name(src->buffer) : "-", what);
    }
    return true;
}

static void ggml_backend_sched_pf_verify(ggml_backend_sched_t sched, int split_id, int ring, int slot, const struct ggml_tensor * input, struct ggml_tensor * input_cpy,
        int64_t lo, int64_t hi, const char * when) {
    static int reported = 0;
    if (reported >= 6) { return; }
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    ggml_backend_t be = sched->backends[split->backend_id];
    if (sched->prefetch_ring[ring].backend) { ggml_backend_synchronize(sched->prefetch_ring[ring].backend); }
    ggml_backend_synchronize(be);
    const size_t esz = input->nb[2];
    const size_t n   = (size_t) (hi - lo) * esz;
    static std::vector<uint8_t> tmp;
    tmp.resize(n);
    ggml_backend_tensor_get(input_cpy, tmp.data(), (size_t) lo * esz, n);   // input_cpy points at the slot (moved back by lo experts for a range slot)
    const uint8_t * ref = (const uint8_t *) input->data + (size_t) lo * esz;
    if (memcmp(tmp.data(), ref, n) == 0) { return; }
    size_t first = 0, n_diff = 0, last = 0;
    for (size_t i = 0; i < n; i++) { if (tmp[i] != ref[i]) { if (n_diff++ == 0) { first = i; } last = i; } }
    reported++;
    const char * base = sched->prefetch_ring[ring].region_buf ? (const char *) ggml_backend_buffer_get_base(sched->prefetch_ring[ring].region_buf) : NULL;
    const char * at   = (const char *) input_cpy->data + (size_t) lo * esz;
    GGML_LOG_WARN("pf-verify: %s split %d on %s: %s experts [%lld,%lld) in slot %d: %zu of %zu bytes differ from the host tensor, from byte %zu (expert %lld + %zu) to byte %zu (expert %lld); "
            "slot at %.1f MiB of its buffer, %.1f MiB long\n", when, split_id, ggml_backend_name(be), input->name, (long long) lo, (long long) hi, slot, n_diff, n, first,
            (long long) (lo + (int64_t) (first / esz)), first % esz, last, (long long) (lo + (int64_t) (last / esz)),
            base ? (double) (at - base) / 1048576.0 : -1.0, (double) n / 1048576.0);
    char hex[160] = ""; char hex2[160] = "";
    for (size_t i = 0; i < 24 && first + i < n; i++) { snprintf(hex + 3*i, 4, "%02x ", tmp[first + i]); snprintf(hex2 + 3*i, 4, "%02x ", ref[first + i]); }
    GGML_LOG_WARN("pf-verify:   slot bytes there: %s\npf-verify:   host bytes:       %s\n", hex, hex2);
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_backend_nvtx_init();
    struct ggml_backend_sched_split * splits = sched->splits;
    g_in_done.clear();   // a new graph: its user inputs are copied again, once per card (ggml_backend_sched_copy_input)

    int prev_backend_id = -1;

    // GGML_SCHED_PROF=1: host-side time per graph averaged over 64 graphs (ported from the fork's scheduler, same line as before the #29943 merge)
    static const bool prof = getenv("GGML_SCHED_PROF") != NULL;
    static int64_t pw_cpu = 0, pw_gpu = 0, pc_cpu = 0, pl_gpu = 0, p_total = 0, p_n = 0, p_splits = 0;
    static int64_t pb_cpu = 0, pb_gpu = 0, pc_tiny = 0, n_tiny = 0;
    const int64_t p_graph_t0 = prof ? ggml_time_us() : 0;

    // GGML_SCHED_TRACE=<file>: one CSV row per split of every prompt-sized graph (GGML_SCHED_TRACE_ALL=1: every graph), research. Where a pass spends its time,
    // split by split, in four parts that are separated by synchronising the card after each of them (so the numbers add up, at the price of serialising the
    // overlap of uploads and compute): gap_us = host time since the previous split ended (launch and scheduling overhead), barrier_us = waiting for the previous
    // card before the inputs may be touched, copy_us = send and receive: the input copies to this card (activations from the previous card, K/V, weights that
    // live in host memory, uploads), compute_us = this card's kernels. Columns: graph,split,backend,tokens,nodes,first,last,inputs,in_bytes,host_w_bytes,gap_us,
    // barrier_us,copy_us,compute_us (first/last = names of the first and last node of the split, which says which layer it is).
    static FILE * tr_file = [] () -> FILE * {
        const char * f = getenv("GGML_SCHED_TRACE");
        if (!f || !*f) { return NULL; }
        FILE * fp = fopen(f, "a");
        if (fp && ftell(fp) == 0) { fprintf(fp, "graph,split,backend,tokens,nodes,first,last,inputs,in_bytes,host_w_bytes,gap_us,barrier_us,copy_us,compute_us\n"); }
        return fp;
    }();
    static const bool tr_all = getenv("GGML_SCHED_TRACE_ALL") != NULL;
    static long tr_graph = 0;
    int64_t tr_tokens = 0;
    if (tr_file && sched->n_splits > 0) {
        const struct ggml_cgraph & lg = splits[sched->n_splits - 1].graph;
        tr_tokens = lg.n_nodes > 0 ? lg.nodes[lg.n_nodes - 1]->ne[1] : 0;
    }
    const bool tracing = tr_file && (tr_all || tr_tokens > 1);
    int64_t tr_prev_end = 0;
    if (tracing) { tr_graph++; }

    // prefetch overlap: a staged expert tensor of a future split (see ggml_backend_sched_prefetch_target). LOOKAHEAD splits ahead of the one
    // computing, so by the time the consumer starts the upload is done and its wait is a no-op; the slot count must exceed the lookahead.
    static const int LOOKAHEAD = 1;
    struct prefetch_pending {
        int            slot       = -1;
        int            ring       = 0;     // index of the card's ring in sched->prefetch_ring
        ggml_tensor *  input      = NULL;
        ggml_tensor *  input_cpy  = NULL;
        ggml_backend_buffer_t saved_buffer = NULL;
        void *         saved_data = NULL;
        ggml_backend_buffer_t slot_buf = NULL;
        int64_t        lo         = 0;     // expert range staged: [lo, hi); hi = n_expert when the node owns them all
        int64_t        hi         = 0;
        bool           redirected = false; // input_cpy points at the slot: from the fire to the launch of its last reader
        int            last_reader = -1;   // the last split that reads input_cpy (this split, or a later one: see cpy_last)
    };
    std::vector<prefetch_pending> lookahead;
    ggml_backend_buffer_t pf_slot_buf [GGML_SCHED_MAX_PF_RINGS][GGML_SCHED_MAX_PREFETCH_SLOTS] = {};
    uint8_t *             pf_slot_data[GGML_SCHED_MAX_PF_RINGS][GGML_SCHED_MAX_PREFETCH_SLOTS] = {};
    const bool pf_on = sched->prefetch_n_slots >= 2 && !sched->callback_eval && sched->n_splits > 1;
    if (pf_on) {
        lookahead.resize(sched->n_splits);
    }
    // fork: the last split that reads each staged copy. Two splits can read one weight through one copy tensor: --moe split-pieces runs a layer's expert
    // matmuls once per token range (one range per card), and the scheduler copies a host weight once per card, for the FIRST split that reads it; the later
    // ones read the same copy and do not list the weight among their inputs, so nothing stages or uploads it for them. The staging slot that fills the copy
    // must stay in place until the last of them has been launched: given back after the first, the later matmul read the copy's own memory in the graph
    // buffer, which nothing had filled (NaN, '!!!!' answers: Qwen3.8-Flash-Next 12k and 60k, every run with split-pieces=1, 2026-10-24). The slot is then
    // held across the splits in between (held_n: fire_ahead counts it as in flight, so the ring never hands it to another upload before it is released).
    std::unordered_map<const ggml_tensor *, int> cpy_last;
    std::unordered_map<int, std::vector<int>>    held_by;   // reader split -> the staged splits whose slots it releases at its launch
    std::vector<int> held_n(sched->n_backends, 0);
    if (pf_on) {
        for (int j = 0; j < sched->n_splits; j++) {
            ggml_tensor * in = NULL, * cpy = NULL;
            if (ggml_backend_sched_prefetch_target(sched, &splits[j], &in, &cpy)) { cpy_last[cpy] = j; }
        }
        if (!cpy_last.empty()) {   // decode graphs stage nothing: no scan
            for (int j = 0; j < sched->n_splits; j++) {
                const struct ggml_cgraph & g = splits[j].graph;
                for (int k = 0; k < g.n_nodes; k++) {
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        const ggml_tensor * src = g.nodes[k]->src[s];
                        if (src == NULL) { continue; }
                        auto it = cpy_last.find(src);
                        if (it != cpy_last.end() && it->second < j) { it->second = j; }
                    }
                }
            }
        }
    }
    // GGML_PF_STATS=1: one line per prompt graph on stderr, how many splits were staged by the prefetch ring and why the others were not (see prefetch_target for 1..5)
    static const bool pf_stats = getenv("GGML_PF_STATS") != NULL;
    int pfs_fired = 0, pfs_why[7] = {0}, pfs_card = 0, pfs_dev = 0, pfs_prep = 0, pfs_off = pf_on ? 0 : 1;
    double pfs_res_b = 0.0, pfs_up_b = 0.0;   // staged bytes copied on the card from its expert cache / uploaded from host memory
    // GGML_PF_TRACE=1: for graphs of 300+ splits, one line per staged copy (split, card, expert range, microseconds since the graph started, the split being launched at that moment)
    // and one per launched split (card, first node), to read the issue order of both cards' copies against an nsys timeline.
    static const bool pf_trace = getenv("GGML_PF_TRACE") && atoi(getenv("GGML_PF_TRACE")) == 1;
    const int64_t pft_t0 = ggml_time_us();
    int pft_cur = -1;
    // Release of a staging slot. A slot may be overwritten once the matmul that read it has finished: an event recorded on the card's compute stream behind that
    // split, which the next upload into the slot waits for. Recorded right behind the split, the upload started the moment the matmul ended, and at the end of a
    // card's expert chain that is exactly when the layer's hand-off needs the same link: the copies of one card run one after the other, so the helper's partial sum
    // stood behind the owner's own upload (nsys, OLMoE 3.4k tokens, x16 owner + x4 helper: 2.0 ms per layer with no kernel on the owner, and the three uploads of a
    // layer were three pieces spread over 20 ms). Rule, for any number of cards: if a copy between GPUs that touches this card (as source or destination) comes
    // later in the graph, the slot is released behind THAT copy, together with the card's other slots waiting for it. The uploads of a layer's three tensors then
    // run as one block right after the hand-off (for the owner: under its dense part; for a helper: while it has nothing to compute), and never in front of it.
    // A card without a later hand-off (one GPU, the last layer) releases behind the split as before. GGML_PF_RELEASE=split restores that everywhere.
    static const bool pf_release_split = getenv("GGML_PF_RELEASE") && strcmp(getenv("GGML_PF_RELEASE"), "split") == 0;
    struct pend_free { int ring; int slot; };
    std::vector<std::vector<pend_free>> pf_pend(sched->n_backends);
    std::vector<std::vector<int>> xnext;   // xnext[b][j]: the first split >= j whose input copies cross between GPUs and touch backend b; n_splits when none
    int  pft_issued  = -1;                 // the last split whose input copies have been queued
    bool pf_deferred = false;              // set by fire_prefetch: the upload waits for a hand-off that releases its slot
    sched->xcopy_touched = 0;
    if (pf_on && !pf_release_split) {
        const auto is_gpu = [&](int b) { return ggml_backend_dev_type(ggml_backend_get_device(sched->backends[b])) == GGML_BACKEND_DEVICE_TYPE_GPU; };
        std::vector<uint32_t> xm(sched->n_splits, 0);
        for (int j = 0; j < sched->n_splits; j++) {
            const struct ggml_backend_sched_split * sj = &splits[j];
            for (int i = 0; i < sj->n_inputs; i++) {
                struct ggml_tensor * in = sj->inputs[i];
                if (ggml_backend_sched_is_user_input(in) || ggml_backend_sched_is_host_weight(in)) { continue; }
                const int sb = tensor_backend_id(in);
                if (sb >= 0 && sb != sj->backend_id && is_gpu(sb) && is_gpu(sj->backend_id)) { xm[j] |= (1u << sb) | (1u << sj->backend_id); }
            }
        }
        xnext.assign(sched->n_backends, std::vector<int>(sched->n_splits + 2, sched->n_splits));
        for (int b = 0; b < sched->n_backends; b++) {
            for (int j = sched->n_splits - 1; j >= 0; j--) { xnext[b][j] = (xm[j] >> b & 1) ? j : xnext[b][j + 1]; }
        }
    }
    const auto release_slots = [&](int b) {
        for (const pend_free & q : pf_pend[b]) { ggml_backend_event_record(sched->prefetch_ring[q.ring].free[q.slot], sched->backends[b]); }
        pf_pend[b].clear();
    };
    // the slot of staged split j is given back: its last reader, split `at` on card b, has been launched (the kernels captured the slot's address); the copy
    // tensor points at its own memory again
    const auto pf_finish = [&](int j, int at, int b) {
        prefetch_pending & q = lookahead[j];
        if (pf_release_split || xnext.empty() || xnext[b][at + 1] >= sched->n_splits) {
            ggml_backend_event_record(sched->prefetch_ring[q.ring].free[q.slot], sched->backends[b]);
        } else {
            pf_pend[b].push_back({ q.ring, q.slot });   // behind this card's next hand-off, see "release of a staging slot"
        }
        sched->prefetch_ring[q.ring].used[q.slot] = true;
        q.input_cpy->buffer = q.saved_buffer;
        q.input_cpy->data   = q.saved_data;
        q.redirected = false;
    };
    // card stamps (see ggml_sched_card_stat): the passes whose expert chains are measured, every split on a GPU; set up below, before the first staging
    // upload is issued (fire_prefetch stamps the upload stream too)
    int cs_set = -1;
    std::vector<int> cs_idx(sched->n_splits, -1);
    const auto fire_prefetch = [&](int target_id) {
        char nvtx_pf[120] = "";
        if (g_nvtx_on) { snprintf(nvtx_pf, sizeof nvtx_pf, "PREFETCH issue split %d %s", target_id, target_id < sched->n_splits ? ggml_backend_name(sched->backends[splits[target_id].backend_id]) : "-"); }
        nvtx_scope nvtx_pfs(nvtx_pf);
        if (!pf_on || target_id >= sched->n_splits || lookahead[target_id].slot != -1 || sched->prefetch_n_slots < 2) {
            return;
        }
        struct ggml_backend_sched_split * s = &splits[target_id];
        ggml_tensor * input = NULL, * input_cpy = NULL;
        int why = 0;
        if (!ggml_backend_sched_prefetch_target(sched, s, &input, &input_cpy, &why)) {
            pfs_why[why]++;
            static const bool pf_dump = getenv("GGML_PF_STATS") && atoi(getenv("GGML_PF_STATS")) >= 2;
            static int dumped = 0;
            if (pf_dump && (why == 2 || why == 3) && s->graph.n_nodes > 2 && dumped < 14 && sched->n_splits > 100 && sched->n_splits < 160) {
                dumped++;
                GGML_LOG_WARN("pf-dump: split %d on %s, why %d, %d nodes, %d inputs; nodes:", target_id, ggml_backend_name(sched->backends[s->backend_id]), why, s->graph.n_nodes, s->n_inputs);
                for (int i = 0; i < s->graph.n_nodes && i < 6; i++) {
                    GGML_LOG_WARN(" [%s %s]", ggml_op_name(s->graph.nodes[i]->op), s->graph.nodes[i]->name);
                }
                GGML_LOG_WARN(" inputs:");
                for (int i = 0; i < s->n_inputs && i < 6; i++) {
                    GGML_LOG_WARN(" [%s]", s->inputs[i]->name);
                }
                GGML_LOG_WARN("\n");
            }
            return;
        }
        // fork: two splits that read the same weight through the same copy tensor (the expert matmul of every token range of a dense split pass, --moe
        // split-pieces) are never staged at the same time. The copy's data pointer points at the slot from the fire to the split's launch, and a second fire in
        // between moved it under the first: that split ran on the other's slot while it was still being uploaded, the other on the unstaged buffer ('!!!!'
        // answers, Qwen3.8-Flash-Next 60k, 2026-10-24). The later split is fired once the earlier one has launched (fire_ahead runs after every launch).
        for (int j = 0; j < target_id; ++j) {
            if (lookahead[j].redirected && lookahead[j].input_cpy == input_cpy) {
                return;
            }
        }
        ggml_backend_t s_backend = sched->backends[s->backend_id];
        // The ring of this split's card (created on first use): its own stream, events and slots, and the staging blocks the expert cache lent on THIS card.
        struct sched_pf_ring * R = ggml_backend_sched_pf_ring_for(sched, s_backend->device);
        if (R == NULL || R->n_slots < 2) {
            pfs_dev++;
            return;
        }
        const int ri = (int) (R - sched->prefetch_ring);
        const size_t need = ggml_backend_sched_prefetch_bytes(sched, s, input);
        if (!ggml_backend_sched_prefetch_prepare(sched, R, s_backend, need, R->n_slots, pf_slot_buf[ri], pf_slot_data[ri])) {
            pfs_prep++;
            return;
        }
        {
            // the slot this upload would take is still waiting for its release behind a hand-off of this card: the upload is issued then (fire_ahead comes back
            // after every launch). No hand-off before the target split needs the slot: released now, behind everything queued on the card so far.
            bool waiting = false;
            for (const pend_free & q : pf_pend[s->backend_id]) { if (q.ring == ri && q.slot == R->cur) { waiting = true; } }
            if (waiting) {
                if (!xnext.empty() && xnext[s->backend_id][pft_issued + 1] < target_id) { pf_deferred = true; return; }
                release_slots(s->backend_id);
            }
        }
        pfs_fired++;
        if (pf_trace && sched->n_splits > 300) {
            const ggml_tensor * mmn = s->graph.nodes[0];
            GGML_LOG_WARN("pf-trace: staged split %d card %s experts [%d,%d) issued at %lld us while launching split %d\n", target_id, ggml_backend_name(s_backend),
                    (int) mmn->op_params[8], (int) mmn->op_params[9], (long long) (ggml_time_us() - pft_t0), pft_cur);
        }
        const int slot = R->cur;
        R->cur = (R->cur + 1) % R->n_slots;
        if (R->used[slot]) {
            ggml_backend_event_wait(R->backend, R->free[slot]);   // the compute that read this slot has finished
        }
        // card stats: the upload stream's own time for this split's experts, from behind the slot's release to the ready event (link_us)
        int64_t * ut1_dst = NULL;
        if (cs_set >= 0 && cs_idx[target_id] >= 0 && ggml_backend_sched_pass_detail()) {
            g_stamp_fn(R->backend, &g_card_buf[cs_set][cs_idx[target_id]].ut0);
            ut1_dst = &g_card_buf[cs_set][cs_idx[target_id]].ut1;
        }
        prefetch_pending & p = lookahead[target_id];
        p.slot = slot; p.ring = ri; p.input = input; p.input_cpy = input_cpy; p.slot_buf = pf_slot_buf[ri][slot];
        p.saved_buffer = input_cpy->buffer;
        p.saved_data   = input_cpy->data;
        p.redirected   = true;
        {
            const auto it = cpy_last.find(input_cpy);
            p.last_reader = it != cpy_last.end() ? std::max(it->second, target_id) : target_id;
        }
        input_cpy->buffer = pf_slot_buf[ri][slot];
        input_cpy->data   = pf_slot_data[ri][slot];
        // Only the experts this MUL_MAT_ID owns (--moe prefill-split gives each card an id range, op_params 8 and 9; 0 is the precision, 1 the skip flag, 3 the src1 precision): one copy of that contiguous range,
        // not the whole tensor. The padding after the range is the MMQ rule of sched_copy_experts: the next expert's first bytes must not hold NaNs.
        const ggml_tensor * mm_node = s->graph.nodes[0];
        p.lo = 0; p.hi = input->ne[2];
        if (mm_node->op_params[9] > mm_node->op_params[8]) { p.lo = mm_node->op_params[8]; p.hi = std::min<int64_t>(mm_node->op_params[9], input->ne[2]); }
        // A slot that holds only the range (see ggml_backend_sched_prefetch_bytes) starts with expert `lo`: the tensor's data pointer is moved back by lo experts,
        // so expert id still sits at data + id x expert size. The matmul reads only ids in [lo, hi) (the others are -1), so it never leaves the slot.
        if (p.lo > 0 && need < ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(s_backend), input)) {
            input_cpy->data = pf_slot_data[ri][slot] - (size_t) p.lo*input->nb[2];
        }
        {
            const size_t esz = input->nb[2];
            const size_t off = (size_t) p.lo*esz;
            const size_t pad = p.hi < input->ne[2] ? ggml_backend_sched_expert_pad(sched, s->backend_id, input) : 0;
            // Staging fill from what the card already holds. The card's expert cache keeps experts for generation in its pool (Qwen3.8-Flash-Next on the x16
            // card: 19 GB, about 40% of all experts); a prompt pass used to upload every expert of the range from host memory anyway, those included. Now the
            // experts the cache holds on this card are copied on the card (one kernel for all of them, CUDA: ggml_backend_cuda_copy_slices_fork, else one copy
            // each) and only the runs of missing ones cross the link. Both go on the ring's own stream, so the ready event below covers them, as before. The
            // MMQ padding rule stays: the bytes after the range come from host memory (the run that reaches `hi` carries them), and the tensor's end is zeroed.
            // --moe pf-resident=0 (no query registered) is the old single upload.
            std::vector<const void *> where;
            ggml_backend_buffer_t where_buf = NULL;
            int n_res = 0;
            if (g_moe_where_fn) {
                where.assign((size_t) (p.hi - p.lo), NULL);
                n_res = g_moe_where_fn(input, (int32_t) p.lo, (int32_t) p.hi, s_backend->device, where.data(), &where_buf, g_moe_where_ud);
            }
            if (n_res <= 0 || where_buf == NULL) {
                ggml_backend_tensor_set_async(R->backend, input_cpy, (const char *) input->data + off, off, (size_t) (p.hi - p.lo)*esz + pad);
                pfs_up_b += (double) (p.hi - p.lo)*esz;
                ggml_backend_sched_upload_add(s_backend->device, (uint64_t) (p.hi - p.lo)*esz);
            } else {
                typedef bool (*copy_slices_t)(ggml_backend_t, const void * const *, void * const *, size_t, size_t);
                static std::mutex slices_mu;
                static ggml_backend_dev_t slices_dev[16] = {};
                static copy_slices_t      slices_fn [16] = {};
                copy_slices_t copy_slices = NULL;
                {
                    std::lock_guard<std::mutex> lk(slices_mu);
                    int k = 0;
                    while (k < 16 && slices_dev[k] != NULL && slices_dev[k] != s_backend->device) { k++; }
                    if (k < 16 && slices_dev[k] == NULL) {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(s_backend->device);
                        slices_dev[k] = s_backend->device;
                        slices_fn[k]  = reg ? (copy_slices_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_slices_fork") : NULL;
                    }
                    copy_slices = k < 16 ? slices_fn[k] : NULL;
                }
                // GGML_PF_SLICES=1 (test): the one-launch slice kernel. Off by default: it takes its pointer arrays from the backend's CUDA pool, and with the
                // compute alias that pool is served from lent blocks the compute stream uses at the same time (Qwen3.8-Flash-Next perplexity, chunk 2: illegal
                // memory access). One device copy per expert instead (a few microseconds of host time each).
                static const bool use_slices = getenv("GGML_PF_SLICES") && atoi(getenv("GGML_PF_SLICES")) == 1;
                if (!use_slices) { copy_slices = NULL; }
                std::vector<const void *> srcs;
                std::vector<void *> dsts;
                int64_t run0 = -1;
                for (int64_t id = p.lo; id <= p.hi; ++id) {
                    const bool res = id < p.hi && where[(size_t) (id - p.lo)] != NULL;
                    if (id < p.hi && !res) {
                        if (run0 < 0) { run0 = id; }
                        continue;
                    }
                    if (run0 >= 0) {   // a run of missing experts ends here: one upload (with the padding when it reaches the range's end)
                        const size_t o = (size_t) run0*esz;
                        const size_t n = (size_t) (id - run0)*esz + (id == p.hi ? pad : 0);
                        ggml_backend_tensor_set_async(R->backend, input_cpy, (const char *) input->data + o, o, n);
                        pfs_up_b += (double) (id - run0)*esz;
                        ggml_backend_sched_upload_add(s_backend->device, (uint64_t) (id - run0)*esz);
                        run0 = -1;
                    } else if (id == p.hi && pad > 0) {   // the range ended with a resident expert: the padding bytes still come from host memory
                        ggml_backend_tensor_set_async(R->backend, input_cpy, (const char *) input->data + (size_t) p.hi*esz, (size_t) p.hi*esz, pad);
                    }
                    if (res) {
                        srcs.push_back(where[(size_t) (id - p.lo)]);
                        dsts.push_back((uint8_t *) input_cpy->data + (size_t) id*esz);
                    }
                }
                if (!srcs.empty() && !(copy_slices && copy_slices(R->backend, srcs.data(), dsts.data(), srcs.size(), esz))) {
                    for (size_t i = 0; i < srcs.size(); ++i) {
                        ggml_tensor sv = {}, dv = {};
                        sv.type = GGML_TYPE_I8; sv.buffer = where_buf; sv.data = (void *) srcs[i];
                        sv.ne[0] = (int64_t) esz; sv.ne[1] = sv.ne[2] = sv.ne[3] = 1; sv.nb[0] = 1; sv.nb[1] = sv.nb[2] = sv.nb[3] = esz;
                        dv = sv; dv.buffer = input_cpy->buffer; dv.data = dsts[i];
                        ggml_backend_tensor_copy_async(R->backend, R->backend, &sv, &dv);
                    }
                }
                pfs_res_b += (double) srcs.size()*esz;
            }
            if (p.hi == input->ne[2]) {
                ggml_backend_tensor_zero_tail_fork(R->backend, input_cpy);   // the range ends at the tensor's end: zeros behind it (see there)
            }
        }
        if (ut1_dst) { g_stamp_fn(R->backend, ut1_dst); }
        ggml_backend_event_record(R->ready[slot], R->backend);
    };
    // The lookahead is counted PER CARD, not in splits: after split k is launched, every card keeps its next PF_PER_CARD staged expert splits in flight (fired, not yet
    // launched). Counted in splits (LOOKAHEAD) it fired only split k+2, and in the multi-GPU prompt split, where the splits of a layer alternate between the cards, the
    // other card's upload for the coming layer was issued only when the scheduler reached that card's split: nsys, GLM 8k, split: the x4 card's upload started 60-110 ms after
    // the x16's and its compute waited for it (copy, then compute, no overlap), the cards' kernels ran at the same time for 0.47 s of 17 s. With one card the two rules
    // are the same (k+1 was fired earlier, k+2 now). PF_WINDOW bounds the scan for splits that are no prefetch target.
    // In flight per card: one less than the card's staging slots (a slot is reused only when the matmul that read it has finished), at least LOOKAHEAD + 1. With 3 slots
    // that is 2, fewer than one layer's 3 expert tensors (gate, up, down): the next layer's copies could start only when the current layer's matmuls ran, which on the slow
    // card is at the end of the layer (nsys, OLMoE split: the x4 card prefetched nothing ahead). With 6 slots (two layers) the next layer's copies start right after the
    // current layer's, limited by the copy engine only. PF_WINDOW: the scan reaches over a whole layer of splits in the multi-GPU split (about 10 per layer).
    const int PF_PER_CARD = std::max(LOOKAHEAD + 1, sched->prefetch_n_slots - 1);
    static const int PF_WINDOW = 64;
    const auto fire_ahead = [&](int k) {
        std::vector<int> pend(held_n);   // slots held for a later reader of their copy are in flight too
        std::vector<char> held(sched->n_backends, 0);   // this card's next upload waits for a slot release: nothing later of the card may overtake it (the ring is first in, first out)
        for (int j = k + 1; j < sched->n_splits && j <= k + PF_WINDOW; j++) {
            const int b = splits[j].backend_id;
            if (lookahead[j].slot != -1) { pend[b]++; continue; }
            // a card whose ring got fewer slots for this graph (the lent region holds fewer, see prefetch_prepare) keeps fewer uploads in flight: one less than its slots,
            // or the ring would hand an upload the slot of a split that has not run yet
            const struct sched_pf_ring * Rb = ggml_backend_sched_pf_ring_for(sched, sched->backends[b]->device);
            const int lim = Rb ? std::min(PF_PER_CARD, std::max(LOOKAHEAD + 1, Rb->n_slots - 1)) : PF_PER_CARD;
            if (pend[b] >= lim || held[b]) { continue; }
            pf_deferred = false;
            fire_prefetch(j);
            if (pf_deferred) { held[b] = 1; }
            if (lookahead[j].slot != -1) { pend[b]++; }
        }
    };
    // Stamps of the expert chains (see ggml_sched_chain_stat): the expert matmul splits of the multi-GPU prompt split are grouped by (layer, card); the layer is
    // the number at the end of the MUL_MAT_ID's name ("ffn_moe_gate-12"). Only when the previous measurement has left the cards.
    int ch_set = -1, ch_n = 0, ch_ended = 0;
    std::vector<int>  ch_first(sched->n_splits, -1), ch_last(sched->n_splits, -1);
    std::vector<int>  ch_tail(sched->n_backends, -1);   // the chain that ended last on this card, until anything else is launched on it
    std::vector<char> ch_out;
    sched->xcopy_src = 0;
    if (g_stamp_fn != NULL && !sched->callback_eval && sched->n_splits > 1) {
        std::vector<struct ggml_sched_chain_stat> found;
        std::vector<int> found_last;
        for (int j = 0; j < sched->n_splits; j++) {
            const struct ggml_backend_sched_split * sj = &splits[j];
            if (sj->graph.n_nodes == 0) { continue; }
            const struct ggml_tensor * n0 = sj->graph.nodes[0];
            if (n0->op != GGML_OP_MUL_MAT_ID || !(n0->op_params[9] > n0->op_params[8])) { continue; }
            const char * dash = strrchr(n0->name, '-');
            if (dash == NULL || dash[1] < '0' || dash[1] > '9') { continue; }
            const int layer = atoi(dash + 1);
            int idx = -1;
            for (size_t i = 0; i < found.size(); i++) {
                if (found[i].layer == layer && found[i].backend == sched->backends[sj->backend_id] && found[i].lo == n0->op_params[8]) { idx = (int) i; }
            }
            if (idx < 0) {
                struct ggml_sched_chain_stat e = {};
                e.layer = layer; e.backend = sched->backends[sj->backend_id]; e.lo = n0->op_params[8]; e.hi = n0->op_params[9];
                e.n_tokens = n0->src[2] ? n0->src[2]->ne[1] : 0;
                idx = (int) found.size();
                found.push_back(e); found_last.push_back(j);
                ch_first[j] = idx;
            }
            found_last[idx] = j;
        }
        if (!found.empty() && found.size() <= GGML_SCHED_MAX_CHAIN_STATS) {
            std::lock_guard<std::mutex> lk(g_chain_mu);
            ggml_backend_sched_chain_roll();
            if (g_chain_live < 0) {
                ch_set = g_chain_done == 0 ? 1 : 0;
                ch_n   = (int) found.size();
                for (int i = 0; i < ch_n; i++) { g_chain_buf[ch_set][i] = found[i]; ch_last[found_last[i]] = i; }
                g_chain_n[ch_set] = ch_n;
                g_chain_live = ch_set;
                ch_out.assign(ch_n, 0);
            }
        }
        if (ch_set < 0) { std::fill(ch_first.begin(), ch_first.end(), -1); }
    }
    // card stamps (see ggml_sched_card_stat): the passes whose expert chains are measured, every split on a GPU
    static const bool card_stats = !(getenv("GGML_SCHED_CARD_STATS") && atoi(getenv("GGML_SCHED_CARD_STATS")) == 0);
    if (card_stats && ch_set >= 0 && sched->n_splits <= GGML_SCHED_MAX_CARD_STAMPS && sched->n_backends <= GGML_SCHED_CARD_MAX_BE) {
        std::lock_guard<std::mutex> lk(g_chain_mu);
        ggml_backend_sched_card_roll();
        if (g_card_live < 0) {
            cs_set = g_card_done == 0 ? 1 : 0;
            int n = 0;
            for (int j = 0; j < sched->n_splits; j++) {
                const struct ggml_backend_sched_split * sj = &splits[j];
                if (sj->graph.n_nodes == 0) { continue; }
                if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[sj->backend_id])) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }   // the stamp is a GPU stream function
                const struct ggml_tensor * n0 = sj->graph.nodes[0];
                struct ggml_sched_card_stamp & c = g_card_buf[cs_set][n];
                c.backend_id = sj->backend_id;
                c.moe = n0->op == GGML_OP_MUL_MAT_ID && n0->op_params[9] > n0->op_params[8] ? 1 : 0;
                c.xin = 0; c.upin = 0;
                c.nn = sj->graph.n_nodes;
                snprintf(c.nf, sizeof c.nf, "%s", ggml_op_name(n0->op)); snprintf(c.nl, sizeof c.nl, "%s", ggml_op_name(sj->graph.nodes[sj->graph.n_nodes - 1]->op)); c.ni[0] = 0;
                { size_t l = strlen(c.nf); snprintf(c.nf + l, sizeof c.nf - l, " %s", n0->name); }
                { const struct ggml_tensor * t = sj->graph.nodes[sj->graph.n_nodes - 1]; size_t l = strlen(c.nl); snprintf(c.nl + l, sizeof c.nl - l, " %s", t->name); }
                // GGML_SCHED_PASS_TRACE_OPS=1: every op of the split, as the name of the tensor it computes (cut to what fits), printed under its timeline line
                c.ops[0] = 0;
                if (getenv("GGML_SCHED_PASS_TRACE_OPS")) {
                    size_t l = 0;
                    for (int k = 0; k < sj->graph.n_nodes && l + 24 < sizeof c.ops; k++) {
                        const struct ggml_tensor * t = sj->graph.nodes[k];
                        const char * nm = t->name[0] ? t->name : ggml_op_name(t->op);
                        l += snprintf(c.ops + l, sizeof c.ops - l, "%.*s ", 22, nm);
                    }
                }
                c.t0 = 0; c.tw = 0; c.t1 = 0; c.ut0 = 0; c.ut1 = 0; c.up_host_us = 0;
                cs_idx[j] = n++;
            }
            for (int b = 0; b < sched->n_backends; b++) { g_card_be[cs_set][b] = sched->backends[b]; }
            g_card_n[cs_set] = n;
            g_up_n[cs_set]   = 0;
            g_card_live = n > 0 ? cs_set : -1;
            if (n == 0) { cs_set = -1; }
        }
    }
    if (pf_on) {
        fire_ahead(-1);   // after the card stamps are set up: the first staging uploads are measured too
    }

    // GGML_SCHED_COPY_AHEAD=0 turns off. Inputs (not weights) of split i+1 that come from other cards are copied BEFORE split i is launched, when split i+1 is on
    // another GPU and does not consume anything split i computes. A cross-card copy waits for an event recorded on the SOURCE card at the moment of the copy, i.e.
    // behind everything queued there by then: the two expert chains of the multi-GPU prompt split (both fed by the same hidden state and id masks from the
    // attention part) were serialised by it, whichever went first (nsys, GLM 8k: the second chain started when the first ended, plus the hop). Copied before the
    // first chain is launched, the event sits right after the attention part and both chains start together.
    static const bool copy_ahead = !(getenv("GGML_SCHED_COPY_AHEAD") && atoi(getenv("GGML_SCHED_COPY_AHEAD")) == 0);
    std::vector<char> ahead_done(sched->n_splits, 0);
    std::vector<char> ahead_w_done(sched->n_splits, 0);   // the same for the host-weight inputs (expert copy callback), see below
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        pft_cur = split_id;
        char nvtx_name[200] = "";
        if (g_nvtx_on && split->graph.n_nodes > 0) { snprintf(nvtx_name, sizeof nvtx_name, "SPLIT %d %s %s %s", split_id, ggml_backend_name(split_backend), ggml_op_name(split->graph.nodes[0]->op), split->graph.nodes[0]->name); }
        nvtx_scope nvtx_split(nvtx_name);
        static const bool pf_trace_all = getenv("GGML_PF_TRACE") && atoi(getenv("GGML_PF_TRACE")) == 2;
        if (pf_trace_all && sched->n_splits >= 40 && split->graph.n_nodes > 0) {   // GGML_PF_TRACE=2: every split launch, to find the splits that take the host long
            const ggml_tensor * n0 = split->graph.nodes[0];
            GGML_LOG_WARN("launch-trace: split %d/%d card %s nodes %d first %s %s inputs %d at %lld us abs %lld\n", split_id, sched->n_splits, ggml_backend_name(split_backend), split->graph.n_nodes,
                    ggml_op_name(n0->op), n0->name, split->n_inputs, (long long) (ggml_time_us() - pft_t0), (long long) ggml_time_us());
        }
        if (pf_trace && sched->n_splits > 300 && split->graph.n_nodes > 0 && split->graph.nodes[0]->op == GGML_OP_MUL_MAT_ID) {
            GGML_LOG_WARN("pf-trace: launch split %d card %s MUL_MAT_ID at %lld us (staged %s)\n", split_id, ggml_backend_name(split_backend), (long long) (ggml_time_us() - pft_t0),
                    (pf_on && lookahead[split_id].slot != -1) ? "yes" : "no");
        }
        const int64_t p_t0 = prof ? ggml_time_us() : 0;
        const int64_t tr_t0 = tracing ? ggml_time_us() : 0;
        const prefetch_pending * pf = (pf_on && lookahead[split_id].slot != -1) ? &lookahead[split_id] : NULL;
        const bool    p_is_cpu = prof && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits.
        // The regions this split can overwrite are those of its own backend's compute buffer (sched->bufts[split_backend_id]), where the
        // allocator may have reused memory the previous split's asynchronous copies are still reading. If no input of the previous split lives in
        // a buffer of that type, nothing it reads can be overwritten here: no barrier. (The #29943 merge took the unconditional barrier, which made
        // every CPU split after a GPU split wait for the GPU: +6.6 ms per GLM decode graph, OLMoE -16% on the RTX 4060 laptop.)
        const auto prev_reads_buffer_of_split_backend = [&]() {
            const struct ggml_backend_sched_split * prev = &splits[split_id - 1];
            // fork: a copy between two GPUs reads its source on the SOURCE card's own stream (CUDA: the read half of the staged copy, or the peer copy, both queued
            // on the source stream; g_xcopy_dst_ordered says so), so this split's kernels on that card are queued behind the read and cannot overwrite the tensor
            // before it has been read: no barrier for such inputs. With the dense split (token ranges on two cards, splits alternating between them) the barrier
            // held the scheduler thread 42.8 s over 746 calls in two GLM-5.3-Flash 8k passes (pf-stats site 5) and made the cards run one after the other.
            const bool ordered = g_xcopy_dst_ordered != NULL && g_xcopy_dst_ordered(sched->backends[split_backend_id], sched->backends[prev->backend_id]);
            for (int j = 0; j < prev->n_inputs; j++) {
                const struct ggml_tensor * in = prev->inputs[j];
                ggml_backend_buffer_t buf = in->view_src ? in->view_src->buffer : in->buffer;
                if (!buf || ggml_backend_buffer_get_type(buf) == sched->bufts[split_backend_id]) {
                    if (buf && ordered) {
                        // GGML_SCHED_XORDER_CHECK=1: does the shortcut ever skip a barrier that matters? A barrier matters when a node of THIS split writes into the
                        // region that the previous split's copy reads (the allocator reused it). Counted and the first few named; no effect on the schedule.
                        static const bool chk = [] { const char * e = getenv("GGML_SCHED_XORDER_CHECK"); return e && atoi(e) != 0; }();
                        if (chk) {
                            static std::atomic<int> n_skipped{0}, n_overlap{0};
                            const uint8_t * a0 = (const uint8_t *) in->data, * a1 = a0 + ggml_nbytes(in);
                            n_skipped++;
                            for (int k = 0; k < split->graph.n_nodes; k++) {
                                const struct ggml_tensor * w = split->graph.nodes[k];
                                const uint8_t * b0 = (const uint8_t *) w->data, * b1 = b0 + ggml_nbytes(w);
                                if (w->data && a0 && a0 < b1 && b0 < a1 && w->buffer == in->buffer) {
                                    if (n_overlap++ < 8) {
                                        GGML_LOG_WARN("sched-check: split %d node %s (%s) overwrites the region of %s that split %d's copy reads (%d of %d skipped barriers overlap)\n",
                                            split_id, w->name, ggml_op_name(w->op), in->name, split_id - 1, n_overlap.load(), n_skipped.load());
                                    }
                                    break;
                                }
                            }
                            if (n_skipped % 2000 == 0) {
                                GGML_LOG_WARN("sched-check: %d skipped barriers, %d with an overlapping write\n", n_skipped.load(), n_overlap.load());
                            }
                        }
                        continue;
                    }
                    return true;
                }
            }
            return false;
        };
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id && prev_reads_buffer_of_split_backend()) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                SYNC_TIMED(5, ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]));
            } else {
                SYNC_TIMED(5, ggml_backend_synchronize(sched->backends[prev_backend_id]));
            }
        }

        const int64_t p_tb = prof ? ggml_time_us() : 0;
        const int64_t tr_tb = tracing ? ggml_time_us() : 0;
        if (prof) {
            (p_is_cpu ? pb_cpu : pb_gpu) += p_tb - p_t0;
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        // (--moe pass-stats: their uploads are measured against this split's card stamp entry, see ggml_sched_up_stamp)
        g_cs_cur_set   = cs_set >= 0 && ggml_backend_sched_pass_detail() ? cs_set : -1;
        g_cs_cur_entry = g_cs_cur_set >= 0 ? cs_idx[split_id] : -1;
        std::vector<ggml_backend_t> d2h_pending;
        NVTX_PUSH("COPY inputs (not weights)");
        for (int input_id = 0; input_id < split->n_inputs && !ahead_done[split_id]; input_id++) {
            if (pf && split->inputs[input_id] == pf->input) {
                continue;   // staged by the prefetch
            }
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], d2h_pending);
            }
        }
        NVTX_POP();
        NVTX_PUSH("COPY weights (expert copy callback)");
        for (int input_id = 0; input_id < split->n_inputs && !ahead_w_done[split_id]; input_id++) {
            if (pf && split->inputs[input_id] == pf->input) {
                continue;
            }
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], d2h_pending);
            }
        }
        NVTX_POP();
        for (ggml_backend_t b : d2h_pending) {
            SYNC_TIMED(4, ggml_backend_synchronize(b));
        }
        if (copy_ahead && !sched->callback_eval && split_id + 1 < sched->n_splits && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            struct ggml_backend_sched_split * nx = &splits[split_id + 1];
            ggml_backend_t nx_backend = sched->backends[nx->backend_id];
            if (nx->backend_id != split_backend_id && !ahead_done[split_id + 1] &&
                    ggml_backend_dev_type(ggml_backend_get_device(nx_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                // Does the next split read something this split computes? An input can be a VIEW of such a tensor (a slice of this split's K handed to the next
                // split's cache write, a reshaped state): the view is another tensor, and its node can lie in the next split's own range, so the tensor it
                // looks into is checked too. Compared by the input alone, such a copy was issued before this split had run and the other card read what the
                // memory held before: Qwen3.8-Flash-Next with the dense split, perplexity 32.9 where one card gives 1.125 (1.126 with this check, and with
                // GGML_SCHED_COPY_AHEAD=0).
                bool dep = false;
                for (int i = 0; i < nx->n_inputs && !dep; i++) {
                    const struct ggml_tensor * in  = nx->inputs[i];
                    const struct ggml_tensor * src = in->view_src;
                    for (int j = 0; j < split->graph.n_nodes; j++) {
                        const struct ggml_tensor * nd = split->graph.nodes[j];
                        if (nd == in || (src != NULL && (nd == src || nd->view_src == src))) { dep = true; break; }
                    }
                }
                if (!dep) {
                    g_cs_cur_entry = g_cs_cur_set >= 0 ? cs_idx[split_id + 1] : -1;   // these copies feed the next split
                    std::vector<ggml_backend_t> nx_d2h;
                    for (int i = 0; i < nx->n_inputs; i++) {
                        if (!ggml_backend_sched_is_host_weight(nx->inputs[i])) {
                            ggml_backend_sched_copy_input(sched, nx, nx->inputs[i], nx_d2h);
                        }
                    }
                    for (ggml_backend_t b2 : nx_d2h) { SYNC_TIMED(6, ggml_backend_synchronize(b2)); }
                    ahead_done[split_id + 1] = 1;
                    // The host-weight inputs of the next split (the experts: the copy callback reads the split's ids, which are copied above, and queues the copies of the
                    // experts it uses on THAT card's stream) are issued now too, before this split is launched. Left to the next split's own turn, the two cards' expert
                    // copies of a layer were issued one after the other (nsys, GLM 8k, split: the x4's copy ended and only then its compute began, 60-110 ms after the x16's
                    // upload started). A split staged by the prefetch ring keeps that path. GGML_SCHED_COPY_AHEAD_W=0 turns this part off.
                    static const bool ahead_w = !(getenv("GGML_SCHED_COPY_AHEAD_W") && atoi(getenv("GGML_SCHED_COPY_AHEAD_W")) == 0);
                    if (ahead_w && !(pf_on && lookahead[split_id + 1].slot != -1)) {
                        for (int i = 0; i < nx->n_inputs; i++) {
                            if (ggml_backend_sched_is_host_weight(nx->inputs[i])) {
                                ggml_backend_sched_copy_input(sched, nx, nx->inputs[i], nx_d2h);
                            }
                        }
                        for (ggml_backend_t b2 : nx_d2h) { SYNC_TIMED(6, ggml_backend_synchronize(b2)); }
                        ahead_w_done[split_id + 1] = 1;
                    }
                    g_cs_cur_entry = g_cs_cur_set >= 0 ? cs_idx[split_id] : -1;
                }
            }
        }
        {
            // t_out: this card's half of a copy between GPUs was just queued right behind its expert chain: its partial sum leaves for the layer's owner
            const uint32_t xsrc = sched->xcopy_src;
            sched->xcopy_src = 0;
            for (int b = 0; b < sched->n_backends && ch_set >= 0; b++) {
                if ((xsrc >> b & 1) && ch_tail[b] >= 0 && !ch_out[ch_tail[b]]) {
                    g_stamp_fn(sched->backends[b], &g_chain_buf[ch_set][ch_tail[b]].t_out);
                    ch_out[ch_tail[b]] = 1;
                }
            }
        }
        if (pf_on && !pf_release_split) {
            pft_issued = std::max(pft_issued, split_id + ((split_id + 1 < sched->n_splits && ahead_done[split_id + 1]) ? 1 : 0));
            const uint32_t touched = sched->xcopy_touched;
            sched->xcopy_touched = 0;
            for (int b = 0; b < sched->n_backends; b++) {
                // Released behind the LAST copy of a run INTO this card: the adds of several helpers' partial sums follow each other on the owner, each its own
                // split with one incoming copy. A hand-off of the next layer is no such run (this card's dense part or idle time lies in between): the release
                // must not wait for it, or the uploads would start with the next expert phase instead of under the dense part.
                const int nj = pft_issued + 1;
                const bool run_on = nj < sched->n_splits && splits[nj].backend_id == b && xnext[b][nj] == nj;
                if ((touched >> b & 1) && !pf_pend[b].empty() && !run_on) { release_slots(b); }
            }
        }
        if (tracing) { SYNC_TIMED(7, ggml_backend_synchronize(split_backend)); }   // the copies have arrived
        const int64_t p_t1 = prof ? ggml_time_us() : 0;
        const int64_t tr_tc = tracing ? ggml_time_us() : 0;

        if (g_split_cb && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            g_split_cb(split_backend, g_split_ud);
        }
        ch_tail[split_backend_id] = -1;
        if (ch_set >= 0 && ch_first[split_id] >= 0) {
            g_stamp_fn(split_backend, &g_chain_buf[ch_set][ch_first[split_id]].t_start);
        }
        if (cs_set >= 0 && cs_idx[split_id] >= 0) {
            // what the gap in front of this split waits for (see ggml_sched_card_stat): another GPU's result, or a host weight uploaded on this stream
            struct ggml_sched_card_stamp & cst = g_card_buf[cs_set][cs_idx[split_id]];
            for (int i = 0; i < split->n_inputs && ggml_backend_sched_pass_detail(); i++) {
                ggml_tensor * in = split->inputs[i];
                if (ggml_backend_sched_is_user_input(in)) { continue; }
                if (ggml_backend_sched_is_host_weight(in)) { cst.upin |= !(pf && in == pf->input); continue; }
                const int ib = tensor_backend_id(in);
                if (ib >= 0 && ib != split_backend_id &&
                        ggml_backend_dev_type(ggml_backend_get_device(sched->backends[ib])) == GGML_BACKEND_DEVICE_TYPE_GPU) { if (!cst.xin) { snprintf(cst.ni, sizeof cst.ni, "%s", in->name); } cst.xin = 1; }
            }
            g_stamp_fn(split_backend, &cst.t0);
        }
        if (pf) {
            // the staged tensor is uploaded when the second stream's event has fired; the compute stream waits for it on the device, not the host
            ggml_backend_event_wait(split_backend, sched->prefetch_ring[pf->ring].ready[pf->slot]);
            if (cs_set >= 0 && cs_idx[split_id] >= 0 && ggml_backend_sched_pass_detail()) { g_stamp_fn(split_backend, &g_card_buf[cs_set][cs_idx[split_id]].tw); }   // the wait for the upload ends here
            // GGML_PF_VERIFY=1 (debug, slow): both streams are drained and the staged range is read back and compared with the host tensor, before the split
            // runs (did the upload arrive whole, did anything write into the slot since) and again after it. The first difference is reported: split, card,
            // tensor, range, slot, the expert and the byte in it, and the slot's place in its buffer.
            static const bool pf_verify = getenv("GGML_PF_VERIFY") != NULL;
            if (pf_verify) {
                ggml_backend_sched_pf_verify(sched, split_id, pf->ring, pf->slot, pf->input, pf->input_cpy, pf->lo, pf->hi, "before");
            }
            // let the expert cache see (count) the experts of this batch and keep some of them, device to device from the staged copy
            // (queued after the wait, on the compute stream); every expert is staged, the callbacks decide which ones are worth a block
            ggml_backend_dev_t pdev = ggml_backend_get_device(split_backend);
            // The expert cache's preheat from the prompt (usage counts and the adoption copies of the staged experts) is optional and costs host time per expert and tensor and a
            // device-to-device copy of every adopted expert (GLM 8k: 24 GB of D2D copies in a pass). It is OFF here for now; GGML_PF_COUNT=1 turns it back on (measured: +11.4 decode
            // hit points on GLM 12k with all slots adopting).
            static const bool pf_count = getenv("GGML_PF_COUNT") && atoi(getenv("GGML_PF_COUNT")) == 1;
            const int64_t n_exp = pf_count ? pf->hi : pf->lo;   // the staged range; nothing when the preheat is off
            const size_t  exp_sz = pf->input->nb[2];
            for (int64_t id = pf->lo; id < n_exp; ++id) {
                const void * dd = NULL; ggml_backend_buffer_t db = NULL;
                (void) ggml_backend_moe_src(pf->input, (int32_t) id, pdev, &dd, &db);   // counts the use; the staged copy already has the expert
                void * dst_data = NULL; ggml_backend_buffer_t dst_buf = NULL;
                if (ggml_backend_moe_fill(pf->input, (int32_t) id, pdev, &dst_data, &dst_buf)) {
                    ggml_tensor s = {}, d = {};
                    s.type = GGML_TYPE_I8; s.buffer = pf->slot_buf; s.data = (uint8_t *) pf->input_cpy->data + (size_t) id*exp_sz;
                    s.ne[0] = (int64_t) exp_sz; s.ne[1] = s.ne[2] = s.ne[3] = 1; s.nb[0] = 1; s.nb[1] = s.nb[2] = s.nb[3] = exp_sz;
                    d = s; d.buffer = dst_buf; d.data = dst_data;
                    ggml_backend_tensor_copy_async(split_backend, split_backend, &s, &d);
                }
            }
        }
        if (!sched->callback_eval) {
            static const int nan_mode = getenv("GGML_SCHED_NAN_CHECK") ? atoi(getenv("GGML_SCHED_NAN_CHECK")) : 0;
            static long nan_graph2 = 0;
            enum ggml_status ec = GGML_STATUS_SUCCESS;
            if (nan_mode == 2 && sched->n_splits >= 30) {
                if (split_id == 0) { nan_graph2++; }
                for (int nj = 0; nj < split->graph.n_nodes && ec == GGML_STATUS_SUCCESS; nj++) {
                    struct ggml_cgraph gv = ggml_graph_view(&split->graph, nj, nj + 1);
                    ec = ggml_backend_graph_compute_async(split_backend, &gv);
                    ggml_backend_sched_nan_check_node(sched, split_id, nj, nan_graph2);
                }
            } else {
                NVTX_PUSH("LAUNCH graph_compute_async"); ec = ggml_backend_graph_compute_async(split_backend, &split->graph); NVTX_POP();
            }
            if (cs_set >= 0 && cs_idx[split_id] >= 0) {
                if (ec == GGML_STATUS_SUCCESS) {
                    g_stamp_fn(split_backend, &g_card_buf[cs_set][cs_idx[split_id]].t1);
                } else {
                    std::lock_guard<std::mutex> lk(g_chain_mu);
                    if (g_card_live == cs_set) { g_card_live = -1; }   // a failed pass is never complete: drop it, or no later pass would be measured
                    cs_set = -1;
                }
            }
            if (ch_set >= 0 && ch_last[split_id] >= 0 && ec == GGML_STATUS_SUCCESS) {
                g_stamp_fn(split_backend, &g_chain_buf[ch_set][ch_last[split_id]].t_end);
                ch_tail[split_backend_id] = ch_last[split_id];
                ch_ended++;
            }
            if (pf && getenv("GGML_PF_VERIFY") != NULL) {
                ggml_backend_sched_pf_verify(sched, split_id, pf->ring, pf->slot, pf->input, pf->input_cpy, pf->lo, pf->hi, "after");
            }
            if (pf) {
                // the kernels have captured the slot address at launch: the staging copy may point back, the slot is free once this split is done; unless a
                // later split reads the same copy (see cpy_last): then both stay until that split has been launched
                if (pf->last_reader > split_id) {
                    held_by[pf->last_reader].push_back(split_id);
                    held_n[split_backend_id]++;
                } else {
                    pf_finish(split_id, split_id, split_backend_id);
                }
            }
            if (!held_by.empty()) {
                auto it = held_by.find(split_id);
                if (it != held_by.end()) {   // this split was the last reader of earlier staged copies
                    for (int j : it->second) { pf_finish(j, split_id, split_backend_id); held_n[split_backend_id]--; }
                    held_by.erase(it);
                }
            }
            if (pf_on) {
                NVTX_PUSH("FIRE_AHEAD prefetch for next splits");
                fire_ahead(split_id);
                NVTX_POP();
            }
            if (ec != GGML_STATUS_SUCCESS) {
                for (auto & kv : held_by) { for (int j : kv.second) { pf_finish(j, split_id, splits[j].backend_id); } }
                held_by.clear();
                for (int b = 0; b < sched->n_backends; b++) { release_slots(b); }
                return ec;
            }
            {
                static const bool nan_check = getenv("GGML_SCHED_NAN_CHECK") && atoi(getenv("GGML_SCHED_NAN_CHECK")) == 1;
                static long nan_graph = 0;
                if (nan_check && sched->n_splits >= 30) {
                    if (split_id == 0) { nan_graph++; }
                    ggml_backend_sched_nan_check(sched, split_id, nan_graph);
                }
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        if (tracing) {
            SYNC_TIMED(8, ggml_backend_synchronize(split_backend));   // the kernels are done
            const int64_t tr_td = ggml_time_us();
            size_t in_bytes = 0, hw_bytes = 0;
            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                (ggml_backend_sched_is_host_weight(split->inputs[input_id]) ? hw_bytes : in_bytes) += ggml_nbytes(split->inputs[input_id]);
            }
            const struct ggml_cgraph & g = split->graph;
            fprintf(tr_file, "%ld,%d,%s,%lld,%d,%s,%s,%d,%zu,%zu,%lld,%lld,%lld,%lld\n", tr_graph, split_id, ggml_backend_name(split_backend), (long long) tr_tokens, g.n_nodes,
                    g.n_nodes > 0 ? g.nodes[0]->name : "", g.n_nodes > 0 ? g.nodes[g.n_nodes - 1]->name : "", split->n_inputs, in_bytes, hw_bytes,
                    (long long) (tr_prev_end ? tr_t0 - tr_prev_end : 0), (long long) (tr_tb - tr_t0), (long long) (tr_tc - tr_tb), (long long) (tr_td - tr_tc));
            fflush(tr_file);
            tr_prev_end = tr_td;
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        if (prof) {
            const int64_t p_t2 = ggml_time_us();
            (p_is_cpu ? pw_cpu : pw_gpu) += p_t1 - p_t0;
            (p_is_cpu ? pc_cpu : pl_gpu) += p_t2 - p_t1;
            if (p_is_cpu && split->graph.n_nodes <= 2) {
                pc_tiny += p_t2 - p_t1;
                n_tiny++;
            }
        }

        prev_backend_id = split_backend_id;
    }

    if (prof && sched->n_splits > 8) {
        p_total  += ggml_time_us() - p_graph_t0;
        p_splits += sched->n_splits;
        if (++p_n == 64) {
            GGML_LOG_WARN("sched-prof: per graph %.2f ms (%.0f splits): wait before CPU %.2f (barrier %.2f, input copy %.2f), CPU compute %.2f (tiny splits %.1f x %.1f us), wait before GPU %.2f (barrier %.2f, input copy %.2f), GPU launch %.2f ms\n",
                    p_total/1e3/p_n, (double) p_splits/p_n,
                    pw_cpu/1e3/p_n, pb_cpu/1e3/p_n, (pw_cpu - pb_cpu)/1e3/p_n,
                    pc_cpu/1e3/p_n, (double) n_tiny/p_n, n_tiny ? (double) pc_tiny/n_tiny : 0.0,
                    pw_gpu/1e3/p_n, pb_gpu/1e3/p_n, (pw_gpu - pb_gpu)/1e3/p_n, pl_gpu/1e3/p_n);
            pw_cpu = pw_gpu = pc_cpu = pl_gpu = p_total = p_n = p_splits = 0;
            pb_cpu = pb_gpu = pc_tiny = n_tiny = 0;
        }
    }

    for (auto & kv : held_by) {   // a held slot whose reader never ran (cannot happen: every reader is a split of this graph): given back behind its card's work
        for (int j : kv.second) { pf_finish(j, sched->n_splits - 1, splits[j].backend_id); }
    }
    held_by.clear();
    g_cs_cur_set = -1;
    for (int b = 0; b < sched->n_backends; b++) { release_slots(b); }   // slots still waiting for a release (the next graph waits for these events)
    if (ch_set >= 0 && ch_ended != ch_n) {
        std::lock_guard<std::mutex> lk(g_chain_mu);
        if (g_chain_live == ch_set) { g_chain_live = -1; }
    }

    if (pf_stats && sched->n_splits >= 30) {
        const struct ggml_cgraph & lg = splits[sched->n_splits - 1].graph;
        const long long n_tok = lg.n_nodes > 0 ? (long long) lg.nodes[lg.n_nodes - 1]->ne[1] : 0;
        GGML_LOG_WARN("pf-stats (%lld tokens): staged %.2f GB, %.2f GB of it copied on the card from its expert cache, %.2f GB uploaded\n", n_tok,
                (pfs_res_b + pfs_up_b) / 1e9, pfs_res_b / 1e9, pfs_up_b / 1e9);
        GGML_LOG_WARN("pf-stats (%lld tokens, %d slots, prefetch %s): %d splits, prefetch fired for %d; not staged: no inputs %d, first node not MUL_MAT_ID %d, weight not an input %d, "
                "weight not host-resident %d, few ids %d, range part of the card split %d, region on another card %d, staging stream on another card %d, no slots %d\n",
                n_tok, sched->prefetch_n_slots, pf_on ? "on" : "OFF", sched->n_splits, pfs_fired, pfs_why[1], pfs_why[2], pfs_why[3], pfs_why[4], pfs_why[5], pfs_why[6],
                pfs_card, pfs_dev, pfs_prep);
        GGML_LOG_WARN("pf-stats sync (cumulative host ms / calls): copy_input-own-card %lld/%lld, cross-card-fallback %lld/%lld, expert-ids-read %lld/%lld, d2h %lld/%lld, prev-card %lld/%lld, copy-ahead %lld/%lld, trace %lld/%lld, eval-cb %lld/%lld\n",
                g_sync_us[1].load()/1000, g_sync_n[1].load(), g_sync_us[2].load()/1000, g_sync_n[2].load(), g_sync_us[3].load()/1000, g_sync_n[3].load(), g_sync_us[4].load()/1000, g_sync_n[4].load(),
                g_sync_us[5].load()/1000, g_sync_n[5].load(), g_sync_us[6].load()/1000, g_sync_n[6].load(), g_sync_us[7].load()/1000, g_sync_n[7].load(), g_sync_us[8].load()/1000, g_sync_n[8].load());
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    // At most one split per node is the proof-safe bound, but it costs ~21 KB per node (30 inputs x 2 tensors per split, plus the id arrays): at the 575k-node graph
    // of a 14336-token micro-batch that is 12 GB, committed at once. Linux overcommits it unseen; Windows refuses ("GGML_ASSERT(ctx->mem_buffer != NULL)" in ggml_init
    // right after the weights loaded, an 8 GB laptop with 15 GiB of RAM left, any micro-batch from ~12000 tokens). Real graphs have a few hundred to a few thousand
    // splits (one per backend change), so the bound is capped and checked where a split is added.
    const size_t ggml_sched_max_splits = std::min<size_t>(graph_size, 16384);
    sched->max_splits = ggml_sched_max_splits;
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        if (sched->tail_ev[b]) {
            ggml_backend_event_free(sched->tail_ev[b]);
        }
    }
    for (int r = 0; r < sched->prefetch_n_rings; r++) {
        struct sched_pf_ring * R = &sched->prefetch_ring[r];
        if (R->backend) {
            ggml_backend_synchronize(R->backend);
        }
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            ggml_backend_event_free(R->ready[i]);
            ggml_backend_event_free(R->free[i]);
            if (R->buf[i]) {
                ggml_backend_buffer_free(R->buf[i]);
            }
        }
        if (R->backend) {
            ggml_backend_free(R->backend);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

// fork: prefetch overlap. n_slots >= 2 turns the staging of full expert tensors (during prompt batches, while the previous split computes) on; 0 = off
void ggml_backend_sched_set_prefetch_experts_slots(ggml_backend_sched_t sched, int n_slots) {
    sched->prefetch_n_slots = std::max(0, std::min(n_slots, GGML_SCHED_MAX_PREFETCH_SLOTS));
    for (int r = 0; r < sched->prefetch_n_rings; r++) {
        sched->prefetch_ring[r].n_slots = sched->prefetch_n_slots;
        sched->prefetch_ring[r].cur = 0;
    }
}

int ggml_backend_sched_get_prefetch_experts_slots(ggml_backend_sched_t sched) {
    return sched->prefetch_n_slots;
}

// fork: bytes of one staging slot for the current graph (the largest staged expert tensor, padded), 0 when nothing is staged
size_t ggml_backend_sched_prefetch_slot_bytes(ggml_backend_sched_t sched) {
    if (sched->prefetch_n_slots < 2) {
        return 0;
    }
    return GGML_PAD(ggml_backend_sched_prefetch_scan(sched, -1), 256);
}

size_t ggml_backend_sched_prefetch_slot_bytes_dev(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    if (sched->prefetch_n_slots < 2) {
        return 0;
    }
    size_t best = 0;
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_get_device(sched->backends[i]) == dev) {
            best = std::max(best, ggml_backend_sched_prefetch_scan(sched, i));
        }
    }
    return best ? GGML_PAD(best, 256) : 0;
}

// fork: the staging slots of the card `dev` are ranges of `buffer` the host owns, from offset, capacity bytes (the expert cache lends the blocks of that card's pool);
// NULL buffer = allocate. One call per card; the ring of the card is created here if it does not exist yet.
void ggml_backend_sched_set_prefetch_region_dev(ggml_backend_sched_t sched, ggml_backend_dev_t dev, ggml_backend_buffer_t buffer, size_t offset, size_t capacity, bool only) {
    struct sched_pf_ring * R = ggml_backend_sched_pf_ring_for(sched, dev);
    if (R == NULL) { return; }
    R->region_buf  = buffer;
    R->region_off  = offset;
    R->region_cap  = buffer ? capacity : 0;
    R->region_only = only;
}

// the former single-card call: the region goes to the ring of the buffer's own device (every ring when the buffer is NULL)
void ggml_backend_sched_set_prefetch_region(ggml_backend_sched_t sched, ggml_backend_buffer_t buffer, size_t offset, size_t capacity, bool only) {
    if (buffer == NULL) {
        for (int r = 0; r < sched->prefetch_n_rings; r++) {
            sched->prefetch_ring[r].region_buf = NULL; sched->prefetch_ring[r].region_cap = 0; sched->prefetch_ring[r].region_only = only;
        }
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
    ggml_backend_sched_set_prefetch_region_dev(sched, dev, buffer, offset, capacity, only);
}

// fork: the compute graphs of `backend` are placed in `buffer` (from offset, at most capacity bytes) instead of a buffer the scheduler owns;
// its own buffer is freed now. Used to put the compute buffer inside the expert-cache pool (the blocks it covers are blocked while it is in use)
void ggml_backend_sched_set_external_buffer(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_buffer_t buffer, size_t offset, size_t capacity) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            ggml_gallocr_set_external_buffer(sched->galloc, i, buffer, offset, capacity);
            return;
        }
    }
}

// fork: bytes of the backend's compute buffer the current graph uses (what the graph allocator placed, from the start of the buffer)
size_t ggml_backend_sched_graph_peak(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return ggml_gallocr_get_graph_peak(sched->galloc, i);
        }
    }
    return 0;
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARGMAX:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    if (tensor->buffer != NULL) { GGML_LOG_ERROR("%s: tensor '%s' (op %s, data %p) already has a buffer (%s)\n", __func__, tensor->name, ggml_op_name(tensor->op), tensor->data, ggml_backend_buffer_name(tensor->buffer)); }
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

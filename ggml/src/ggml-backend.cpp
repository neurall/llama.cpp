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
#include <unordered_map>
#include <vector>

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

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
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

static ggml_backend_split_cb_t g_split_cb = NULL;
static void *                  g_split_ud = NULL;

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

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
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
#define GGML_SCHED_MAX_PREFETCH_SLOTS 4
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

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
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
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
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

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
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
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
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

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

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

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
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
static void ggml_backend_nvtx_init() {
    static bool done = false;
    if (done) { return; }
    done = true;
    if (!(getenv("GGML_NVTX") && atoi(getenv("GGML_NVTX")) == 1)) { return; }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) { return; }
    g_nvtx_push = (void (*)(const char *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nvtx_push_fork");
    g_nvtx_pop  = (void (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nvtx_pop_fork");
    g_nvtx_on = g_nvtx_push && g_nvtx_pop;
}
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

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input, std::vector<ggml_backend_t> & d2h_pending) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            SYNC_TIMED_NAMED(1, ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        } else {
            SYNC_TIMED_NAMED(1, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        }
        NVTX_PUSH("TENSOR_COPY user input %s", input->name); ggml_backend_tensor_copy(input, input_cpy); NVTX_POP();
        return;
    }

    // wait for the split backend to finish using the input before overwriting it
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } else {
        // no copy events (one input copy, no pipeline parallelism): the destination card's whole stream is drained here, on the host, before every input copy
        SYNC_TIMED_NAMED(9, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
    }

    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
        return;
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

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        SYNC_TIMED_NAMED(2, ggml_backend_synchronize(input_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            SYNC_TIMED_NAMED(2, ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        } else {
            SYNC_TIMED_NAMED(2, ggml_backend_synchronize(split_backend), input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend));
        }
        NVTX_PUSH("TENSOR_COPY sync-copy %s %s->%s", input->name, ggml_backend_name(input_backend), ggml_backend_name(split_backend)); const int64_t _tc = ggml_time_us(); ggml_backend_tensor_copy(input, input_cpy); NVTX_POP(); ggml_backend_sched_sync_stat_add(2, ggml_time_us() - _tc);
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
        if (cand->flags & GGML_TENSOR_FLAG_INPUT) {
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

// bytes of the largest staged expert tensor (as the device reads it: padded rows)
static size_t ggml_backend_sched_prefetch_scan(ggml_backend_sched_t sched, int backend_id) {
    size_t max_size = 0;
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * s = &sched->splits[split_id];
        if (backend_id >= 0 && s->backend_id != backend_id) {
            continue;
        }
        ggml_tensor * in = NULL, * cpy = NULL;
        if (ggml_backend_sched_prefetch_target(sched, s, &in, &cpy)) {
            max_size = std::max(max_size, ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(sched->backends[s->backend_id]), in));
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
    size = GGML_PAD(std::max(size, ggml_backend_sched_prefetch_scan(sched, -1)), 256);
    sched->prefetch_slot_bytes = size;

    // slots inside a buffer the host owns (the expert cache lends the blocks): no allocation here
    if (R->region_buf != NULL && R->region_cap >= (size_t) n * size) {
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(R->region_buf) + R->region_off;
        for (int i = 0; i < n; i++) {
            slot_buf[i]  = R->region_buf;
            slot_data[i] = base + (size_t) i * size;
        }
        return true;
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

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_backend_nvtx_init();
    struct ggml_backend_sched_split * splits = sched->splits;

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
    };
    std::vector<prefetch_pending> lookahead;
    ggml_backend_buffer_t pf_slot_buf [GGML_SCHED_MAX_PF_RINGS][GGML_SCHED_MAX_PREFETCH_SLOTS] = {};
    uint8_t *             pf_slot_data[GGML_SCHED_MAX_PF_RINGS][GGML_SCHED_MAX_PREFETCH_SLOTS] = {};
    const bool pf_on = sched->prefetch_n_slots >= 2 && !sched->callback_eval && sched->n_splits > 1;
    if (pf_on) {
        lookahead.resize(sched->n_splits);
    }
    // GGML_PF_STATS=1: one line per prompt graph on stderr, how many splits were staged by the prefetch ring and why the others were not (see prefetch_target for 1..5)
    static const bool pf_stats = getenv("GGML_PF_STATS") != NULL;
    int pfs_fired = 0, pfs_why[7] = {0}, pfs_card = 0, pfs_dev = 0, pfs_prep = 0, pfs_off = pf_on ? 0 : 1;
    // GGML_PF_TRACE=1: for graphs of 300+ splits, one line per staged copy (split, card, expert range, microseconds since the graph started, the split being launched at that moment)
    // and one per launched split (card, first node), to read the issue order of both cards' copies against an nsys timeline.
    static const bool pf_trace = getenv("GGML_PF_TRACE") && atoi(getenv("GGML_PF_TRACE")) == 1;
    const int64_t pft_t0 = ggml_time_us();
    int pft_cur = -1;
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
        ggml_backend_t s_backend = sched->backends[s->backend_id];
        // The ring of this split's card (created on first use): its own stream, events and slots, and the staging blocks the expert cache lent on THIS card.
        struct sched_pf_ring * R = ggml_backend_sched_pf_ring_for(sched, s_backend->device);
        if (R == NULL || R->n_slots < 2) {
            pfs_dev++;
            return;
        }
        const int ri = (int) (R - sched->prefetch_ring);
        const size_t need = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(s_backend), input);
        if (!ggml_backend_sched_prefetch_prepare(sched, R, s_backend, need, R->n_slots, pf_slot_buf[ri], pf_slot_data[ri])) {
            pfs_prep++;
            return;
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
        prefetch_pending & p = lookahead[target_id];
        p.slot = slot; p.ring = ri; p.input = input; p.input_cpy = input_cpy; p.slot_buf = pf_slot_buf[ri][slot];
        p.saved_buffer = input_cpy->buffer;
        p.saved_data   = input_cpy->data;
        input_cpy->buffer = pf_slot_buf[ri][slot];
        input_cpy->data   = pf_slot_data[ri][slot];
        // Only the experts this MUL_MAT_ID owns (--moe prefill-split gives each card an id range, op_params 8 and 9; 0 is the precision, 1 the skip flag, 3 the src1 precision): one copy of that contiguous range,
        // not the whole tensor. The padding after the range is the MMQ rule of sched_copy_experts: the next expert's first bytes must not hold NaNs.
        const ggml_tensor * mm_node = s->graph.nodes[0];
        p.lo = 0; p.hi = input->ne[2];
        if (mm_node->op_params[9] > mm_node->op_params[8]) { p.lo = mm_node->op_params[8]; p.hi = std::min<int64_t>(mm_node->op_params[9], input->ne[2]); }
        {
            const size_t esz = input->nb[2];
            const size_t off = (size_t) p.lo*esz;
            const size_t pad = p.hi < input->ne[2] ? std::min<size_t>(esz, 512) : 0;
            ggml_backend_tensor_set_async(R->backend, input_cpy, (const char *) input->data + off, off, (size_t) (p.hi - p.lo)*esz + pad);
        }
        ggml_backend_event_record(R->ready[slot], R->backend);
    };
    // The lookahead is counted PER CARD, not in splits: after split k is launched, every card keeps its next PF_PER_CARD staged expert splits in flight (fired, not yet
    // launched). Counted in splits (LOOKAHEAD) it fired only split k+2, and in the multi-GPU prompt split, where the splits of a layer alternate between the cards, the
    // other card's upload for the coming layer was issued only when the scheduler reached that card's split: nsys, GLM 8k, split: the x4 card's upload started 60-110 ms after
    // the x16's and its compute waited for it (copy, then compute, no overlap), the cards' kernels ran at the same time for 0.47 s of 17 s. With one card the two rules
    // are the same (k+1 was fired earlier, k+2 now). PF_WINDOW bounds the scan for splits that are no prefetch target.
    static const int PF_PER_CARD = LOOKAHEAD + 1, PF_WINDOW = 24;
    const auto fire_ahead = [&](int k) {
        std::vector<int> pend(sched->n_backends, 0);
        for (int j = k + 1; j < sched->n_splits && j <= k + PF_WINDOW; j++) {
            const int b = splits[j].backend_id;
            if (lookahead[j].slot != -1) { pend[b]++; continue; }
            if (pend[b] >= PF_PER_CARD) { continue; }
            fire_prefetch(j);
            if (lookahead[j].slot != -1) { pend[b]++; }
        }
    };
    if (pf_on) {
        fire_ahead(-1);
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
            for (int j = 0; j < prev->n_inputs; j++) {
                const struct ggml_tensor * in = prev->inputs[j];
                ggml_backend_buffer_t buf = in->view_src ? in->view_src->buffer : in->buffer;
                if (!buf || ggml_backend_buffer_get_type(buf) == sched->bufts[split_backend_id]) {
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
                bool dep = false;   // does the next split read something this split computes?
                for (int i = 0; i < nx->n_inputs && !dep; i++) {
                    for (int j = 0; j < split->graph.n_nodes; j++) {
                        if (split->graph.nodes[j] == nx->inputs[i]) { dep = true; break; }
                    }
                }
                if (!dep) {
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
                }
            }
        }
        if (tracing) { SYNC_TIMED(7, ggml_backend_synchronize(split_backend)); }   // the copies have arrived
        const int64_t p_t1 = prof ? ggml_time_us() : 0;
        const int64_t tr_tc = tracing ? ggml_time_us() : 0;

        if (g_split_cb && ggml_backend_dev_type(ggml_backend_get_device(split_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            g_split_cb(split_backend, g_split_ud);
        }
        if (pf) {
            // the staged tensor is uploaded when the second stream's event has fired; the compute stream waits for it on the device, not the host
            ggml_backend_event_wait(split_backend, sched->prefetch_ring[pf->ring].ready[pf->slot]);
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
            NVTX_PUSH("LAUNCH graph_compute_async"); enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph); NVTX_POP();
            if (pf) {
                // the kernels have captured the slot address at launch: the staging copy may point back, the slot is free once this split is done
                ggml_backend_event_record(sched->prefetch_ring[pf->ring].free[pf->slot], split_backend);
                sched->prefetch_ring[pf->ring].used[pf->slot] = true;
                pf->input_cpy->buffer = pf->saved_buffer;
                pf->input_cpy->data   = pf->saved_data;
            }
            if (pf_on) {
                NVTX_PUSH("FIRE_AHEAD prefetch for next splits");
                fire_ahead(split_id);
                NVTX_POP();
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
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

    if (pf_stats && sched->n_splits >= 30) {
        const struct ggml_cgraph & lg = splits[sched->n_splits - 1].graph;
        const long long n_tok = lg.n_nodes > 0 ? (long long) lg.nodes[lg.n_nodes - 1]->ne[1] : 0;
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

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

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

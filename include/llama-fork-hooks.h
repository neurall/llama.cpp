#pragma once
// fork: points in a request that libllama cannot see itself, called by the server. A header of its own (not llama.h): adding a hook recompiles only
// the files that include it.
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// a request starts (a slot takes a task): the next decode call is its prompt's first, however short (--moe dynhot looks up its topic there). prompt: the
// tokens the topic's key pools, own_text = the request's own text (a chat request's user input, the template left out), else its whole prompt; the server's
// prompt cache may decode only the prompt's tail, so the key never comes from the decoded call alone
LLAMA_API void llama_moe_request_begin_fork(const int32_t * prompt, int32_t n_prompt, bool own_text);

// a request is over (its final response is being sent): the MoE cache writes its per-request files (--moe expids: <state dir>/hotmap/<model>.expids)
LLAMA_API void llama_moe_request_end_fork(void);

#ifdef __cplusplus
}
#endif

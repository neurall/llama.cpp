#!/usr/bin/env python3
"""ernie55_sidecar.py MODEL_DIR HF_DIR: tokenizer.gguf beside the ERNIE 5.5 Flash GGUF converted for Strata (arch ernie55), whose files hold no tokenizer
("it comes from the release") and keep the logit multimax as two tensors. The sidecar holds what upstream PR 31925's converter writes (conversion/ernie5.py:
gpt2 model, pre ernie5, BOS <|ERNIE_MASK|>) from the release's tokenizer.json, a chat template written after the release's encoding_ernie.py (system, user,
assistant, thinking on or off; no tools, no media), and ernie5_5.logit_multimax. src/llama-gguf-compat-fork.h merges it into the model's keys at load."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

# encoding_ernie.py: render() = START system ... END, the messages without their outer newlines, then "\n" START assistant "\n" and the think prefix
CHAT_TEMPLATE = (
    "{%- set think = enable_thinking is not defined or enable_thinking -%}"
    "<|ernie:im_start|>system\n"
    "{%- if messages[0]['role'] == 'system' -%}<ernie:system_setting>\n{{ messages[0]['content'] }}\n</ernie:system_setting>\n{%- endif -%}"
    "<ernie:global_setting>{%- if think %}\nreasoning_effort=high{%- endif %}\n</ernie:global_setting><|ernie:im_end|>"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'user' or (m['role'] == 'system' and not loop.first) -%}\n<|ernie:im_start|>{{ m['role'] }}\n{{ m['content'] }}<|ernie:im_end|>"
    "{%- elif m['role'] == 'assistant' -%}\n<|ernie:im_start|>assistant\n<ernie:think></ernie:think>{{ m['content'] }}<|ernie:im_end|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}\n<|ernie:im_start|>assistant\n{%- if think %}<ernie:think>{%- else %}<ernie:think></ernie:think>{%- endif -%}{%- endif -%}"
)


def vocab(tok_json):
    """tokens by id, their types (1 normal, 3 control, 4 user defined) and the merges, from a HF tokenizer.json"""
    tj = json.load(open(tok_json, encoding="utf-8"))
    v = dict(tj["model"]["vocab"])
    added = {a["content"]: a for a in tj.get("added_tokens", [])}
    for a in added.values():
        v[a["content"]] = a["id"]
    n = max(v.values()) + 1
    toks = [f"[PAD{i}]" for i in range(n)]
    types = [gguf.TokenType.UNUSED] * n
    for t, i in v.items():
        toks[i] = t
        a = added.get(t)
        types[i] = gguf.TokenType.NORMAL if a is None else gguf.TokenType.CONTROL if a.get("special") else gguf.TokenType.USER_DEFINED
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"].get("merges", [])]
    return toks, types, merges


assert CHAT_TEMPLATE.count("{%- for") == 1

if __name__ == "__main__":
    mdir, hdir = sys.argv[1:3]
    toks, types, merges = vocab(os.path.join(hdir, "tokenizer.json"))
    r = gguf.GGUFReader(os.path.join(mdir, "ERNIE-5.5-Flash-IQ3XXS-IQ4XS-00003-of-00003.gguf"))
    mm = {t.name: [float(x) for x in t.data] for t in r.tensors if t.name.startswith("output.multimax_")}
    w = gguf.GGUFWriter(os.path.join(mdir, "tokenizer.gguf"), "ernie5_5")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("ernie5")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(toks.index("<|ERNIE_MASK|>"))
    w.add_eos_token_id(toks.index("<|ernie:im_end|>"))
    w.add_pad_token_id(toks.index("<|ERNIE_PAD|>"))
    w.add_add_bos_token(True)
    w.add_add_sep_token(False)
    w.add_chat_template(CHAT_TEMPLATE)
    w.add_array("ernie5_5.logit_multimax", mm["output.multimax_ranges"] + mm["output.multimax_ts"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"tokenizer.gguf: {len(toks)} tokens, {len(merges)} merges, multimax {mm}")

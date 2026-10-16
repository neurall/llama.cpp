# micro-OLMoE: 2 layers, 8 experts, ff 256, 4 heads, 2048 vocab - weights small enough for the CPU caches (speed tests only, the output is meaningless)
import gguf, numpy as np, sys
src=gguf.GGUFReader('/m/o/32/OLMoE-1B-7B-0924-F32.gguf')
w=gguf.GGUFWriter('/p/bw/data/olmoe-quant/micro-F32.gguf','olmoe')
over={'olmoe.block_count':2,'olmoe.embedding_length':1024,'olmoe.feed_forward_length':512,'olmoe.attention.head_count':8,'olmoe.attention.head_count_kv':8,'olmoe.expert_count':8,'olmoe.expert_used_count':4}
for k,f in src.fields.items():
    if k.startswith('GGUF.') or k.startswith('tokenizer.') or k=='general.architecture': continue
    vt=f.types[0]
    if vt==gguf.GGUFValueType.ARRAY: continue
    v=over.get(k, f.contents())
    w.add_key_value(k, v, vt)
w.add_key_value('tokenizer.ggml.model','none',gguf.GGUFValueType.STRING)
w.add_key_value('olmoe.vocab_size',2048,gguf.GGUFValueType.UINT32)
def add(name,arr): w.add_tensor(name,np.ascontiguousarray(arr).astype(np.float32))
T={t.name:t for t in src.tensors}
d=lambda n: np.array(T[n].data)
add('token_embd.weight', d('token_embd.weight')[:2048,:1024])
add('output.weight', d('output.weight')[:2048,:1024])
add('output_norm.weight', d('output_norm.weight')[:1024])
for l in range(2):
    p=f'blk.{l}.'
    add(p+'attn_norm.weight', d(p+'attn_norm.weight')[:1024]); add(p+'ffn_norm.weight', d(p+'ffn_norm.weight')[:1024])
    add(p+'attn_q.weight', d(p+'attn_q.weight')[:1024,:1024]); add(p+'attn_k.weight', d(p+'attn_k.weight')[:1024,:1024]); add(p+'attn_v.weight', d(p+'attn_v.weight')[:1024,:1024])
    add(p+'attn_output.weight', d(p+'attn_output.weight')[:1024,:1024])
    add(p+'attn_q_norm.weight', d(p+'attn_q_norm.weight')[:1024]); add(p+'attn_k_norm.weight', d(p+'attn_k_norm.weight')[:1024])
    add(p+'ffn_gate_inp.weight', d(p+'ffn_gate_inp.weight')[:8,:1024])
    add(p+'ffn_gate_exps.weight', d(p+'ffn_gate_exps.weight')[:8,:512,:1024]); add(p+'ffn_up_exps.weight', d(p+'ffn_up_exps.weight')[:8,:512,:1024])
    add(p+'ffn_down_exps.weight', d(p+'ffn_down_exps.weight')[:8,:1024,:512])
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('written')

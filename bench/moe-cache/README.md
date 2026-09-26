# Benchmark prompts for the MoE expert cache

The inputs behind the numbers in the main README, frozen so anyone can reproduce them.

- `src_12k.cpp`: 12,302 tokens of llama.cpp source (the "long" 12k test).
- `src_128k.cpp`: 125,257 tokens of llama.cpp source (the 128k test).
- `run.py`: sends a test to a running `llama-server` and prints prompt and decode speed.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf -c 131072   # -c only needed for 128k
python3 bench/moe-cache/run.py short    # 1500-token chat reply
python3 bench/moe-cache/run.py 12k
python3 bench/moe-cache/run.py 128k
```

Run each test twice and keep the second: the first request after starting the server also
reads the model from disk. With the cache off (`--moe-expert-cache 0`) the same binary
behaves like stock llama.cpp, for comparison. Token counts are for the GLM-5.3-Flash tokenizer; other models tokenize them differently.

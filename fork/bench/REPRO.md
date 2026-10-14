# Reproducing the README rows

Reproduce one row of the README top table: `repro.py ID` prints the exact command (add `--run` to execute it), `repro.py --md` writes REPRO.md.
Every row is: model file, README cell (run.py test alias), arguments (always only -np 1 and, where it is named, -c), the fork build and the stock build.
Build names are directories under the builds directory of run.py (PERF_BUILDS); fork builds are the dev builds of the commit given in the README, stock builds
come from `run.py build --stock --ref <commit>` (llama.cpp upstream at that build number). The load mode of our builds is --load-mode pin (run.py adds it).

<a id="q2-cpu"></a>
## q2-cpu

PC2 (Ryzen 5 3600), llama-cli -dev none -ngl 0: stock 05af0d2b1 (b11302) against release b11648, qwen36.gguf = Qwen3.6-35B-A3B Q2_0, `cli-t100` (file no longer on disk)

<a id="q36-cpu"></a>
## q36-cpu

PC2 (12 cores, clang, no GPU): `llama-cli -m /m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -dev none -p 'write smallest html tetris game' -n 48` (short) or `-f src_1k.cpp -n 32 -c 4096` (edit); stock = upstream master 37d6d653e built with the same compiler, fork = the b12844 CPU archive (llama-release-b12844-...-bin-ubuntu-x64.tar.gz) and the merged tree

<a id="q2-laptop"></a>
## q2-laptop

W10 laptop (4060 8 GB), llama-cli: stock b5cf8ce02 (b11261) against release b11648, same file, `cli-t100` (file no longer on disk)

<a id="glm-12k"></a>
## glm-12k

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf --builds plan16,stock-b11381-836d57176 --variant "v=-np 1 -c 16384" --variants v -t 12kin_1kout_code_review -n 3
```

<a id="glm-12k-pinned"></a>
## glm-12k-pinned

build `inplace` = branch moe-tail-inplace (tag archive/moe-tail-inplace, commit 5d311d71f, b12375); hand-set arguments, one card; the decode rate of this run was not recorded

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf --builds inplace,stock-b11381-836d57176 --variant "v=-np 1 -dev CUDA1 -c 16384 -b 12416 -ub 12416 -t 6 -tb 6 --cpu-moe --load-mode pin --moe pf-slots=3,borrow=0" --variants v -t 12kin_1kout_code_review -n 3
```

<a id="glm-edit"></a>
## glm-edit

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf --builds plan100,stock-b11452-4fbc76dec --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="glm-game"></a>
## glm-game

completion endpoint, 600 tokens

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf --builds release-b12209,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kout_make_game -n 3
```

<a id="glm35-edit"></a>
## glm35-edit

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf --builds release-b12250,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="glm35-game"></a>
## glm35-game

```
python3 fork/run.py bench --models /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf --builds b11707-49fe4b756,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kout_make_game -n 3
```

<a id="mimo-edit"></a>
## mimo-edit

file removed; the 3-bit file replaces it

```
python3 fork/run.py bench --models MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf (132 GB, no longer on disk) --builds release-b12250,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="mimo-chat"></a>
## mimo-chat

chat endpoint, 1500 tokens

```
python3 fork/run.py bench --models MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf (132 GB, no longer on disk) --builds b11707-49fe4b756,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t chat -n 3
```

<a id="qni4-edit"></a>
## qni4-edit

```
python3 fork/run.py bench --models /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --builds release-b12364-6846d4803,stock-b11452-4fbc76dec --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="qni4-12k"></a>
## qni4-12k

```
python3 fork/run.py bench --models /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --builds b11707-49fe4b756,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 12kin_1kout_code_review -n 3
```

<a id="qni4-chat"></a>
## qni4-chat

chat endpoint, 1500 tokens

```
python3 fork/run.py bench --models /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --builds b11365-49db9ee,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t chat -n 3
```

<a id="qni3-edit"></a>
## qni3-edit

ppl: run.py bench -t ppl on the same builds (stock b11574, fork b12844)

```
python3 fork/run.py bench --models /m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --builds release-b12128,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="qni3-game"></a>
## qni3-game

```
python3 fork/run.py bench --models /m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --builds b11707-49fe4b756,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kout_make_game -n 3
```

<a id="qn1-edit"></a>
## qn1-edit

```
python3 fork/run.py bench --models /m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --builds release-b12128,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="q27mtp-edit"></a>
## q27mtp-edit

both sides with MTP

```
python3 fork/run.py bench --models /m/q/5/Qwen3.8-27B-MTP-Q5_K_M.gguf --builds release-b12193,stock-b11381-836d57176 --variant "v=-np 1 --spec-type draft-mtp" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="q27mtp-game"></a>
## q27mtp-game

both sides with MTP

```
python3 fork/run.py bench --models /m/q/5/Qwen3.8-27B-MTP-Q5_K_M.gguf --builds release-b12193,stock-b11381-836d57176 --variant "v=-np 1 --spec-type draft-mtp" --variants v -t 1kout_make_game -n 3
```

<a id="q27nl-edit"></a>
## q27nl-edit

```
python3 fork/run.py bench --models /m/q/4/Qwen3.8-27B-IQ4_NL.gguf --builds b11707-49fe4b756,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kin_1kout_edit_src -n 3
```

<a id="q36-game"></a>
## q36-game

```
python3 fork/run.py bench --models /m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf --builds plan99,stock-b11381-836d57176 --variant "v=-np 1" --variants v -t 1kout_make_game -n 3
```

<a id="olmoe-game"></a>
## olmoe-game

```
python3 fork/run.py bench --models /m/o/8/4/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --builds plan99,stock-b11574-8bb3cf0d2 --variant "v=-np 1" --variants v -t 1kout_make_game -n 3
```

<a id="laptop-17"></a>
## laptop-17

```
N:\build\stock-de7fa0a3c\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 2048 -f N:\bench\prompts\p10.txt
N:\build\smallcard8\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 2048 -f N:\bench\prompts\p10.txt
```

<a id="laptop-12k"></a>
## laptop-12k

```
N:\build\stock-de7fa0a3c\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 14336 -f N:\bench\prompts\p12k.txt
N:\build\smallcard8\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 14336 -f N:\bench\prompts\p12k.txt
```

<a id="laptop-40k"></a>
## laptop-40k

```
N:\build\stock-de7fa0a3c\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 46000 -f N:\bench\prompts\p40k.txt
N:\build\smallcard8\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 46000 -f N:\bench\prompts\p40k.txt
```

<a id="laptop-125k"></a>
## laptop-125k

```
N:\build\stock-de7fa0a3c\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 131072 -f N:\bench\prompts\p125k.txt
N:\build\smallcard8\llama-cli.exe -m N:\m\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c 131072 -f N:\bench\prompts\p125k.txt
```

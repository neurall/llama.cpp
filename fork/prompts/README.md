# Prompts and requests of the benchmark tests

Every prompt the tests send is a plain file in this directory, so you can read exactly what was measured.  
`fork/prompts.ini` names them and says which test uses which; `fork/run.py` reads them from here (`FORK_PROMPTS_DIR` moves the directory).

## Requests sent to a server (the README table's three tests)

Chat requests for `/v1/chat/completions`: one `user` message, `temperature 0`, `top_k 1`, so the answer is repeatable. The server applies the model's own chat template.

| file | test | what it is | prompt size |
|---|---|---|---|
| `req-12k-1k.json` | `pf12k1k` (`in_heavy`) | "Do a code review of the following source file." then `src_12k.cpp`; up to 1024 tokens out | about 12.4k tokens |
| `req-3k-1k-a.json` | `edit4c` (`in_out_heavy`) | `src_3k.cpp`, then "Add a function that counts the lines of this file and prints the count."; 1024 tokens out | about 3k tokens |
| `req-3k-1k-b.json` | `edit4c`, second variant | same source, then "Delete the code in this file that is never used." | about 3k tokens |
| `req-short-1k.json` | `tetrisc` (`out_heavy`) | "generate smallest html tetris game.", 1024 tokens out | about 10 tokens |

## Source texts (frozen snapshots of llama.cpp source)

| file | used for |
|---|---|
| `src_1k.cpp`, `src_3k.cpp`, `src_12k.cpp` | the text inside the 1k / 3k / 12k prompts |
| `src_128k.cpp` | the 128k-token prompt test (`pf128k`) |
| `longsrc.cpp` | the perplexity text (`ppl`, `ppl3`, `pplc`): 2048-token chunks |

## Instruction and chat prompts

| file | used for |
|---|---|
| `prompt-code-review.txt` | the code-review instruction put before a source file (`prefix = code_review`) |
| `prompt-edit-instr.txt` | one edit instruction per line, rotated by repetition (`edit_instr`): add a function, delete unused code, add comments, rename, fix bugs |
| `prompt-edit-warm.txt` | "Explain in two sentences what this file does.": the short warm-up question after a source file |
| `prompt-bug-instr.txt` | the bug-hunt instruction of the long-prompt test (`long1k`): find the bug, explain it, give the fix and a test |
| `prompt-tetris.txt` | the short "tetris" prompt of the `tetrisc` / `t100` tests |
| `prompt-game-template.txt`, `prompt-game-words.txt`, `prompt-game-warm.txt` | the game-generation test (`game4c`): "write smallest html {word} game" with a word from the list per run |
| `prompt-chatv.txt`, `prompt-chat-long.txt` | the chat tests: one short prompt per line (a repeated temperature-0 answer would reuse exactly the cached experts, so runs rotate), and one long chat prompt |

## Your own prompt, no ini edit

Copy a file into this directory and the file name is the test name:

- `prompt-NAME.txt`: the whole file is one chat message, 1024 tokens out. The test is called `NAME`.
- `req-NAME.json`: a request used as it is (remote runs only); the test is called `NAME`.

Then a cell with that test runs on this host or on a rented machine (`-pc`), e.g. a line `mine,qwen36,2,mutex,1024,,` in a list under `fork/runs/` after `echo "Explain what a mutex is in three sentences." > fork/prompts/prompt-mutex.txt`. The server context is sized from the text (characters / 3 plus the answer and some room); for exact tokens or context add a section to `fork/prompts.local.ini` (`[mutex]`, `path =`, `test = chat`, `tokens = `, `ctx = `), which wins over the file name. Files already named in `prompts.ini` are not turned into tests.

## Reading a test's prompt

`fork/prompts.ini` has a section per prompt (`path =`, `in =` for the size shown on the status line) and per test (`alias`, `tokens`, `prefix`, `snippet`). To check a request by hand:

```
python3 -c "import json;d=json.load(open('fork/prompts/req-12k-1k.json'));print(d['messages'][0]['content'][:400])"
```

Always look at the stored **answer** of a run before trusting its speed: a fast run with a garbage answer is a broken run.

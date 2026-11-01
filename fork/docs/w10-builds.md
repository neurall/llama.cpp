W10 builds (cleaned 2026-10-18; the 132 old N:\tmp\*.bat are archived on PC1: /p/bw/data/w10-tmp-bats-20261018.tar)

RELEASE / DEV BUILD (use this): N:\llama.cpp\release-out\winbuild-rel.bat
  CPU backends + llama-server with clang 20 (N:\llvm20), ggml-cuda.dll with MSVC + nvcc 13.4; output N:\build\rel-cuda134-clang
  Flow: git fetch the bundle into N:\llama.cpp, git checkout FETCH_HEAD, run the bat. (run.py build --machine pc3 does the same with winbuild.bat -> build-cuda-dl.)
OTHER recipes still here: release-out\winbuild.bat (all-MSVC build-cuda-dl: only for trees whose CPU code MSVC can compile; the AVX-512 kernels need the clang CPU build, so use winbuild-rel.bat),
  release-out\winpkg*.bat (zip), N:\tmp\wllvm*.bat (VS clang CPU recipe, RUNBOOK step 5).
Everything else was one-off experiments: look in the archive tar only if needed.

HELPER SERVER (always up): stock llama-server with Qwen3.6 on port 8190, scheduled task gensrv (hidden, N:\tmp\gensrv.bat, starts at logon, no time limit, settings in N:\tmp\gensrv_set.ps1). PC1 uses it through [hosts] gen_url:
  run.py gen / --gen writes prompts, and every stored answer is judged by it (run.py judge reads rows afterwards). run.py starts the task over ssh when the server does not answer.
  It holds 6.6 of the 4060's 8 GB: before a GPU test on W10 run schtasks /end /tn gensrv and taskkill /f /im llama-server.exe, with no PC1 rows running (the next judged row starts it again).

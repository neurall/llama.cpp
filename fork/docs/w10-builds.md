W10 builds (cleaned 2026-10-18; the 132 old N:\tmp\*.bat are archived on PC1: /p/bw/data/w10-tmp-bats-20261018.tar)

RELEASE / DEV BUILD (use this): N:\llama.cpp\release-out\winbuild-rel.bat
  CPU backends + llama-server with clang 20 (N:\llvm20), ggml-cuda.dll with MSVC + nvcc 13.4; output N:\build\rel-cuda134-clang
  Flow: git fetch the bundle into N:\llama.cpp, git checkout FETCH_HEAD, run the bat. (run.py build --machine pc3 does the same with winbuild.bat -> build-cuda-dl.)
OTHER recipes still here: release-out\winbuild.bat (all-MSVC build-cuda-dl: only for trees whose CPU code MSVC can compile; the AVX-512 kernels need the clang CPU build, so use winbuild-rel.bat),
  release-out\winpkg*.bat (zip), N:\tmp\wllvm*.bat (VS clang CPU recipe, RUNBOOK step 5).
Everything else was one-off experiments: look in the archive tar only if needed.

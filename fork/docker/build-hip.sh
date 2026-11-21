#!/bin/bash
# Linux HIP module of a release (run.py build --release): libggml-hip.so of tree $1 in image llama-rel-el8-hip with ROCm 10.1 mounted at /opt/rocm-10.1
# (the 13 Windows targets, glibc 2.28 like the rest of the tarball, static libstdc++; no ROCm runtime bundled: users need ROCm 7+)
set -e
W=$1; B=$W/build-linux-hip
T="gfx1030;gfx1031;gfx1032;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
R=/opt/rocm-10.1; export ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$R/lib/llvm/bin:$PATH
G="--gcc-install-dir=/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13"
SLD="-static-libstdc++ -static-libgcc"
cmake -S $W -B $B -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$R/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DCMAKE_C_FLAGS="$G" -DCMAKE_CXX_FLAGS="$G" -DCMAKE_HIP_FLAGS="$G" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
  -DGGML_HIP=ON "-DGPU_TARGETS=$T" "-DAMDGPU_TARGETS=$T" -DGGML_CUDA_FORCE_MMQ=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="$SLD" -DCMAKE_MODULE_LINKER_FLAGS="$SLD" > $B.cfg.log 2>&1 || { tail -20 $B.cfg.log; exit 1; }
cmake --build $B -j $(nproc) --target ggml-hip
ls -la $B/bin/libggml-hip.so && objdump -T $B/bin/libggml-hip.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 && echo HIPMOD_OK

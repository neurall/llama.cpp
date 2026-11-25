#!/bin/bash
# Linux Vulkan module of a release (run.py build --release): libggml-vulkan.so of tree $1 in image llama-rel-el8-vk, same compiler and flags as the CUDA release
# (clang, gcc-toolset-15 headers, static libstdc++, glibc 2.28); glslc runs through glslc-host.sh with the host's glibc (mounted at /hostlib)
set -e
W=$1; B=$W/build-linux-vk
source /opt/rh/gcc-toolset-13/enable && git config --global --add safe.directory '*'
GID=; [ -d /opt/rh/gcc-toolset-15 ] && GID=--gcc-install-dir=/opt/rh/gcc-toolset-15/root/usr/lib/gcc/x86_64-redhat-linux/15
SLD='-static-libstdc++ -static-libgcc'
cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS=$GID -DCMAKE_CXX_FLAGS=$GID \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=OFF -DGGML_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE=/s/glslc-host.sh \
  -DVulkan_LIBRARY=/opt/vulkan/lib/VulkanLoader/lib/libvulkan.so -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="$SLD" -DCMAKE_SHARED_LINKER_FLAGS="$SLD" -DCMAKE_MODULE_LINKER_FLAGS="$SLD" > $B.cfg.log 2>&1 || { tail -20 $B.cfg.log; exit 1; }
cmake --build $B -j ${JOBS:-$(nproc)} --target ggml-vulkan
ls -la $B/bin/libggml-vulkan.so && echo VKMOD_OK

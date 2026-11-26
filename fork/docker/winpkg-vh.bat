@echo off
rem multi-ssd Windows zip: release build N:\build\%1 + ggml-vulkan.dll + ggml-hip.dll (same checkout) + the ROCm 7.13 runtime of N:\pkg\rel7 (mkrel7.bat) and the CUDA and VC runtimes of N:\pkg\rel; zip %2
cd /d N:\llama.cpp
call N:\vs\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1
set VULKAN_SDK=N:\VulkanSDK\1.4.363.0
if exist build-rel-vk\CMakeCache.txt findstr /c:"llvm22" build-rel-vk\CMakeCache.txt >nul || rmdir /s /q build-rel-vk
set PATH=N:\llvm22\bin;%PATH%
if not exist build-rel-vk\CMakeCache.txt cmake -S . -B build-rel-vk -G "Ninja Multi-Config" -D CMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_VULKAN=ON > build-rel-vk.log 2>&1 || (echo CONFIGURE_VK_FAILED & exit /b 1)
cmake --build build-rel-vk --config Release --target ggml-vulkan >> build-rel-vk.log 2>&1 || (echo BUILD_VK_FAILED & exit /b 1)
echo VK_OK
call N:\llama.cpp\release-out\winbuild-hip7r.bat || (echo BUILD_HIP_FAILED & exit /b 1)
echo HIP_OK
set P=N:\tmp\pkgvh
if exist %P% rmdir /s /q %P%
mkdir %P%
xcopy /q /y /e N:\build\%1\* %P%\ >nul
copy /y N:\llama.cpp\build-rel-vk\bin\Release\ggml-vulkan.dll %P%\ >nul
copy /y N:\build\rel-hip7r\ggml-hip.dll %P%\ >nul
for %%F in (amdhip64_7.dll amd_comgr0713.dll hiprtc07013.dll hiprtc-builtins07013.dll rocm_kpack.dll hipblas.dll libhipblaslt.dll rocblas.dll rocsolver.dll) do copy /y N:\pkg\rel7\%%F %P%\ >nul
for %%F in (cublas64_13.dll cublasLt64_13.dll cudart64_13.dll msvcp140.dll vcruntime140.dll vcruntime140_1.dll LICENSE) do copy /y N:\pkg\rel\%%F %P%\ >nul
xcopy /q /y /e /i N:\pkg\rel7\rocblas %P%\rocblas >nul
xcopy /q /y /e /i N:\pkg\rel7\hipblaslt %P%\hipblaslt >nul
cd /d %P%
if exist N:\llama.cpp\release-out\%2 del N:\llama.cpp\release-out\%2
tar -a -cf N:\llama.cpp\release-out\%2 * && echo ZIP_OK
dir N:\llama.cpp\release-out\%2 | find ".zip"
llama-server.exe --version 2>&1 | find "version"

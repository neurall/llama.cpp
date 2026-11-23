@echo off
rem Windows Vulkan + HIP zip of a release (run.py build --release --machine pc3, after the CPU+CUDA build N:\build\%1 of the same checkout): ggml-vulkan.dll (build-rel-vk), ggml-hip.dll
rem (release-out\winbuild-hip.bat), the ROCm 6.4 / CUDA / VC runtimes of N:\pkg\rel; zip %2 in N:\llama.cpp\release-out
cd /d N:\llama.cpp
call N:\vs\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1
cmake --build build-rel-vk --config Release --target ggml-vulkan > build-rel-vk.log 2>&1 || (echo BUILD_VK_FAILED & exit /b 1)
echo VK_OK
call N:\llama.cpp\release-out\winbuild-hip.bat || (echo BUILD_HIP_FAILED & exit /b 1)
echo HIP_OK
set P=N:\tmp\pkgvh
if exist %P% rmdir /s /q %P%
mkdir %P%
xcopy /q /y /e N:\build\%1\* %P%\ >nul
copy /y N:\llama.cpp\build-rel-vk\bin\Release\ggml-vulkan.dll %P%\ >nul
copy /y N:\build\rel-hip64\ggml-hip.dll %P%\ >nul
for %%F in (amdhip64_6.dll amd_comgr_2.dll hipblas.dll hipblaslt.dll rocblas.dll cublas64_13.dll cublasLt64_13.dll cudart64_13.dll msvcp140.dll vcruntime140.dll vcruntime140_1.dll LICENSE) do copy /y N:\pkg\rel\%%F %P%\ >nul
xcopy /q /y /e /i N:\pkg\rel\rocblas %P%\rocblas >nul
xcopy /q /y /e /i N:\pkg\rel\hipblaslt %P%\hipblaslt >nul
cd /d %P%
if exist N:\llama.cpp\release-out\%2 del N:\llama.cpp\release-out\%2
tar -a -cf N:\llama.cpp\release-out\%2 * && echo ZIP_OK
dir N:\llama.cpp\release-out\%2 | find ".zip"
llama-server.exe --version 2>&1 | find "version"

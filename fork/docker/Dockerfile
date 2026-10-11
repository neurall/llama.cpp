# Linux release builder: CUDA 13.2 on Rocky Linux 8 (glibc 2.28: the binaries run on RHEL 8+, Ubuntu 20.04+, Debian 10+ and everything newer)
FROM nvidia/cuda:13.2.0-devel-rockylinux8
RUN dnf install -y --disablerepo=cuda epel-release && dnf install -y --disablerepo=cuda gcc-toolset-13 git make cmake python3 perl-core wget tar xz which diffutils ccache && dnf clean all
# OpenSSL 3 built from source, static, so HTTPS works on every distro with no libssl dependency (EL8 only ships 1.1.1)
ARG OPENSSL=3.0.15
RUN source /opt/rh/gcc-toolset-13/enable && cd /tmp && wget -q https://www.openssl.org/source/openssl-${OPENSSL}.tar.gz && tar xf openssl-${OPENSSL}.tar.gz \
 && cd openssl-${OPENSSL} && ./Configure no-shared no-tests -fPIC --prefix=/opt/openssl --libdir=lib linux-x86_64 && make -j$(nproc) && make install_sw && cd / && rm -rf /tmp/openssl-*
ENV OPENSSL_ROOT_DIR=/opt/openssl
# clang for the C/C++ of every CPU variant (measured 2026-10-03: faster than gcc-toolset-13, which has a code generation regression; same compiler as the Windows release). nvcc keeps gcc-toolset-13 as host compiler.
# clang 21 picks gcc-toolset-15 for headers and static libstdc++/libgcc (newest wins), the build pins it explicitly with --gcc-install-dir
RUN dnf install -y clang llvm libomp libomp-devel gcc-toolset-15-gcc-c++ && dnf clean all

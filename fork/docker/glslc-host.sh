#!/bin/bash
# the SDK's glslc needs glibc 2.34; EL8 has 2.28: run it (build time only) with the host's newer glibc, the module itself still targets 2.28
exec /hostlib/ld-linux-x86-64.so.2 --library-path /hostlib /opt/vulkan/bin/glslc "$@"

#pragma once
// fork: room for host allocations, and what a host allocation that fails does instead of ending the process.
// Windows charges every committed byte (heap, pinned, the pinned weights) against the commit limit, RAM + page file. A box whose page file cannot grow (W10
// 2026-11-26: N: full, page file held at 16 of its 32 GiB, 24.7 of 47.3 GB committed while idle) has 22 GB of free RAM beside little commit, and a free-RAM
// reading pinned the weights up to it: the first decode ended in "uncaught exception: bad allocation". The room is the smaller of free RAM, the commit left and
// the room under a job object's process limit (Windows' cgroup; --moe commit-cap=MiB sets one for tests). The commit left counts what the page file can still
// grow by on its drive (host_pagefile_growth_fork), so a page file that can grow does not shrink the pin.
// The pin of the weights reads it (common_auto_pin, with the free VRAM as the devices' commit), so do the RAM pool and the RAM KV tier; a host allocation that still
// fails inside a decode ends that call with an error (decode_guarded in llama-context.cpp), not the process.
// test: tests/test-hostroom-fork.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>   // PROCESS_MEMORY_COUNTERS_EX; K32GetProcessMemoryInfo is in kernel32 (no psapi.lib)
#pragma comment(lib, "advapi32.lib")   // RegGetValueA (the page files' maximum sizes)
#endif

// the rule: every reading is a ceiling, the room is the lowest (SIZE_MAX: no such ceiling). dev_commit: commit the devices will still take from the same
// limit (Windows WDDM backs the KV arena, the compute buffers and CUDA's modules with commit: under a job cap the 3346 MiB arena's cudaMalloc failed, W10
// 2026-11-26); taken off the commit and the job room, not off the free RAM.
static inline size_t host_room_fork_calc(size_t avail_phys, size_t avail_commit, size_t job_room, size_t dev_commit = 0) {
    const auto less = [](size_t a, size_t b) { return a == SIZE_MAX ? a : a - std::min(a, b); };
    return std::min(avail_phys, std::min(less(avail_commit, dev_commit), less(job_room, dev_commit)));
}

// what a page file can still grow by: up to its maximum, as far as its drive's free space reaches beyond a reserve the system keeps (W10 2026-11-26: the commit
// left read 19.7 GiB with the page file at 16 of its 32 GiB and 112 GB free on N:, so the pin took 8.3 instead of 14.7 GiB: 27.8 instead of 30.4 prompt t/s)
static inline size_t host_pagefile_growth_fork_calc(size_t cur, size_t max, size_t disk_free, size_t reserve) {
    const size_t want = max > cur ? max - cur : 0;
    const size_t can  = disk_free > reserve ? disk_free - reserve : 0;
    return std::min(want, can);
}

#if defined(_WIN32)
// the page files' growth (HKLM ... Memory Management\PagingFiles: "path initial_MB max_MB" per file; a system-managed file, sizes 0 or missing, counts as
// fixed). Reserve: 4 GiB of the drive stay free.
static inline size_t host_pagefile_growth_fork() {
    char buf[4096] = {};
    DWORD n = sizeof(buf) - 2;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management", "PagingFiles", RRF_RT_REG_MULTI_SZ, NULL, buf, &n) != ERROR_SUCCESS) {
        return 0;
    }
    size_t total = 0;
    for (const char * e = buf; *e; e += strlen(e) + 1) {
        char path[MAX_PATH] = {};
        unsigned long long init_mb = 0, max_mb = 0;
        if (sscanf(e, "%259s %llu %llu", path, &init_mb, &max_mb) != 3 || max_mb == 0) { continue; }
        WIN32_FIND_DATAA fd = {};   // the directory entry: GetFileAttributesExA on the open page file fails with a sharing violation (error 32, W10 2026-11-26)
        HANDLE fh = FindFirstFileA(path, &fd);
        if (fh == INVALID_HANDLE_VALUE) { continue; }
        FindClose(fh);
        const size_t cur = ((size_t) fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        char root[4] = { path[0], ':', '\\', 0 };
        ULARGE_INTEGER fr = {};
        if (!GetDiskFreeSpaceExA(root, &fr, NULL, NULL)) { continue; }
        total += host_pagefile_growth_fork_calc(cur, (size_t) max_mb << 20, (size_t) fr.QuadPart, (size_t) 4 << 30);
    }
    return total;
}

// commit this process may still make under its job's process memory limit, SIZE_MAX without one
static inline size_t host_job_room_fork() {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
    if (!QueryInformationJobObject(NULL, JobObjectExtendedLimitInformation, &li, sizeof(li), NULL) ||
        !(li.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_PROCESS_MEMORY)) {
        return SIZE_MAX;
    }
    PROCESS_MEMORY_COUNTERS_EX pm = {};
    pm.cb = sizeof(pm);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pm, sizeof(pm))) {
        return SIZE_MAX;
    }
    return li.ProcessMemoryLimit > pm.PrivateUsage ? (size_t) (li.ProcessMemoryLimit - pm.PrivateUsage) : 0;
}

// free RAM, commit left, job room (one reading of each)
static inline size_t host_room_fork(size_t * avail_phys = nullptr, size_t * avail_commit = nullptr) {
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return 0;
    }
    if (avail_phys)   { *avail_phys   = (size_t) ms.ullAvailPhys; }
    const size_t commit = (size_t) ms.ullAvailPageFile + host_pagefile_growth_fork();   // the limit grows with the page file while its drive has room
    if (avail_commit) { *avail_commit = commit; }
    return host_room_fork_calc((size_t) ms.ullAvailPhys, commit, host_job_room_fork());
}

// --moe commit-cap=MiB (tests): this process may commit at most MiB, through a job object; the low-commit state of a box whose page file cannot grow,
// without holding memory or growing the page file. Returns false when the job could not be set.
static inline bool host_commit_cap_fork(size_t bytes) {
    HANDLE job = CreateJobObjectA(NULL, NULL);
    if (!job) { return false; }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    li.ProcessMemoryLimit = bytes;
    return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li)) && AssignProcessToJobObject(job, GetCurrentProcess());   // the handle stays open: the job lives with the process
}
#endif

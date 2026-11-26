#pragma once
// fork: room for host allocations, and what a host allocation that fails does instead of ending the process.
// Windows charges every committed byte (heap, pinned, the pinned weights) against the commit limit, RAM + page file. A box whose page file cannot grow (W10
// 2026-11-26: N: full, page file held at 16 of its 32 GiB, 24.7 of 47.3 GB committed while idle) has 22 GB of free RAM beside little commit, and a free-RAM
// reading pinned the weights up to it: the first decode ended in "uncaught exception: bad allocation". The room is the smaller of free RAM, the commit left and
// the room under a job object's process limit (Windows' cgroup; --moe commit-cap=MiB sets one for tests).
// test: tests/test-hostroom-fork.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>   // PROCESS_MEMORY_COUNTERS_EX; K32GetProcessMemoryInfo is in kernel32 (no psapi.lib)
#endif

// the rule: every reading is a ceiling, the room is the lowest (SIZE_MAX: no such ceiling)
static inline size_t host_room_fork_calc(size_t avail_phys, size_t avail_commit, size_t job_room) {
    return std::min(avail_phys, std::min(avail_commit, job_room));
}

#if defined(_WIN32)
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
    if (avail_commit) { *avail_commit = (size_t) ms.ullAvailPageFile; }
    return host_room_fork_calc((size_t) ms.ullAvailPhys, (size_t) ms.ullAvailPageFile, host_job_room_fork());
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

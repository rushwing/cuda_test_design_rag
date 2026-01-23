---

# GPU and Product Information

gpu_architecture: Hopper
product_series: Data Center
product: H100

# Module and Version Information

module_name: Unified Memory
cuda_solution_version: "12.2"
cuda_module_version: "12.2.0"

# Story/Ticket Information

story_id: CUDA-UNIMEM-001
story_description: Verify Unified Memory allocation, prefetching, and usage hints
priority: High
test_type: Functional

# Metadata

author: CUDA QA Team
created_date: "2024-03-20"
last_updated: "2024-03-20"
tags:

* unified-memory
* cudaMallocManaged
* prefetch
* mem-advise

---

# Test Case: Unified Memory Management and Optimization

## Overview

This test case validates the behavior of Unified Memory (UM) on Hopper architecture GPUs. It focuses on the allocation of managed memory using `cudaMallocManaged`, manual migration control via `cudaMemPrefetchAsync`, and the application of performance hints using `cudaMemAdvise`. These features allow for a single pointer to be accessible from both the CPU and GPU while providing tools to minimize page fault overhead.

## Requirements Reference

* REQ-UM-001: `cudaMallocManaged` shall allocate memory accessible by any CPU or GPU in the system.
* REQ-UM-002: `cudaMemPrefetchAsync` shall asynchronously move managed memory to the specified device to reduce page faults.
* REQ-UM-003: `cudaMemAdvise` shall provide usage hints (e.g., ReadMostly, SetPreferredLocation) to the CUDA driver to optimize data placement.

## Preconditions

* NVIDIA GPU with Hopper architecture (SM 9.0+) is available.
* System supports Unified Memory (verified via `prop.managedMemory`).
* Concurrent Managed Access is enabled (verified via `prop.concurrentManagedAccess`).
* CUDA Toolkit 12.2 or later is installed.

## Test Data

| Allocation Size | Advice Type | Prefetch Target | Expected Result |
| --- | --- | --- | --- |
| 256 MB | cudaMemAdviseSetReadMostly | GPU Device 0 | Success |
| 1 GB | cudaMemAdviseSetPreferredLocation | CPU (cudaCpuDeviceId) | Success |
| 64 KB | None | GPU Device 0 | Success |

## Test Steps

### Step 1: Allocate Managed Memory

Allocate a buffer that is accessible by both the CPU and the GPU.

```cpp
float *data;
size_t size = 1024 * 1024 * sizeof(float); // 4MB
cudaError_t err = cudaMallocManaged(&data, size);
ASSERT_EQ(err, cudaSuccess) << "Failed to allocate managed memory";

```

### Step 2: Apply Memory Advice

Provide a hint to the driver that the GPU will primarily read this data to optimize caching.

```cpp
int deviceId;
cudaGetDevice(&deviceId);

err = cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);
ASSERT_EQ(err, cudaSuccess) << "Failed to set ReadMostly advice";

```

### Step 3: Prefetch Data to GPU

Asynchronously migrate the memory to the GPU before launching a kernel to prevent on-demand page faulting.

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

err = cudaMemPrefetchAsync(data, size, deviceId, stream);
ASSERT_EQ(err, cudaSuccess) << "Prefetch to GPU failed";

```

### Step 4: Access Memory on GPU and CPU

Launch a kernel to modify the data on the GPU, then synchronize and read it from the CPU.

```cpp
// GPU Access
int threadsPerBlock = 256;
int blocksPerGrid = (1024 * 1024 + threadsPerBlock - 1) / threadsPerBlock;
simpleKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(data);

// Wait for GPU to finish
cudaStreamSynchronize(stream);

// CPU Access
for (int i = 0; i < 10; i++) {
    ASSERT_GT(data[i], 0.0f) << "Data not updated by GPU";
}

```

### Step 5: Prefetch Data Back to CPU

Explicitly move the data back to the CPU using the `cudaCpuDeviceId` constant.

```cpp
err = cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
ASSERT_EQ(err, cudaSuccess) << "Prefetch to CPU failed";
cudaStreamSynchronize(stream);

```

### Step 6: Cleanup

Free the managed memory and destroy the stream.

```cpp
cudaFree(data);
cudaStreamDestroy(stream);

```

## Expected Results

| Condition | Expected Outcome |
| --- | --- |
| Allocation | Non-null pointer; returns `cudaSuccess`. |
| GPU Execution | Kernel completes without illegal memory access errors. |
| Prefetching | Timing should show reduced execution time compared to on-demand migration. |
| Coherency | CPU sees the updates made by the GPU after synchronization. |

## Error Handling

* If `cudaMallocManaged` returns `cudaErrorNotSupported`, verify that the OS and GPU driver support UM.
* Check for `cudaErrorInvalidValue` if using `cudaMemPrefetchAsync` with an invalid device ID or size.
* Use `cuda-memcheck` or `compute-sanitizer` to detect race conditions where both CPU and GPU access the same managed memory simultaneously without synchronization.

## Related Test Cases

* CUDA-STRM-001: Stream Creation and Inter-Stream Synchronization.
* CUDA-UNIMEM-002: Managed Memory with Multi-GPU Peer-to-Peer access.
* CUDA-MEM-001: Device Memory Allocation with cudaMalloc.

## Notes

* On Hopper GPUs, Unified Memory performance is enhanced by the Address Translation Service (ATS) in certain system configurations (e.g., Grace Hopper Superchip).
* Memory prefetching is a hint; the driver may choose to ignore it under heavy memory pressure.
* `cudaMemAdviseSetPreferredLocation` does not force an immediate migration but influences where pages are placed during the next fault.

---


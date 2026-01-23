---
gpu_architecture: Ampere, Hopper
product_series: GeForce RTX 30, Data Center
product: RTX 3080, H100
module_name: Memory Management
cuda_solution_version: "12.0, 12.2"
cuda_module_version: "12.0.1, 12.2.0"
story_id: CUDA-MEM-001, CUDA-MEM-006
story_description: Verify device memory allocation, deallocation, pitched memory, managed memory, and async transfers
priority: High
test_type: Functional
author: CUDA QA Team
created_date: "2024-01-15"
last_updated: "2024-07-20"
tags:
  - memory
  - cudaMalloc
  - cudaFree
  - cudaMallocPitch
  - cudaMallocManaged
  - cudaMemcpyAsync
  - allocation
---

# Memory Management Test Suite

This document contains test cases for CUDA Memory Management module covering basic allocation/deallocation and advanced memory operations.

---

# Test Case 1: Device Memory Allocation with cudaMalloc (CUDA-MEM-001)

## Overview

This test case verifies that `cudaMalloc` correctly allocates device memory on Ampere architecture GPUs. The test covers various allocation sizes from small buffers (1KB) to large allocations approaching device memory limits (4GB). Proper error handling and memory alignment are also validated.

## Requirements Reference

- REQ-MEM-001: Device memory allocation must succeed for valid size requests
- REQ-MEM-002: cudaMalloc must return cudaErrorMemoryAllocation for requests exceeding available memory
- REQ-MEM-003: Allocated memory must be aligned to at least 256 bytes

## Preconditions

- NVIDIA GPU with Ampere architecture (SM 8.0+) is available
- CUDA driver version 525.0 or later is installed
- Minimum 8GB device memory available
- No other CUDA contexts consuming significant device memory
- System is not under memory pressure

## Test Data

| Allocation Size | Expected Result | Notes |
|-----------------|-----------------|-------|
| 1 KB | Success | Minimum practical allocation |
| 1 MB | Success | Common buffer size |
| 256 MB | Success | Medium allocation |
| 1 GB | Success | Large allocation |
| 4 GB | Success/Fail | Depends on available memory |
| 16 GB | Fail | Exceeds typical device memory |

## Test Steps

### Step 1: Initialize CUDA Context
Initialize the CUDA runtime and verify device availability. Query device properties to confirm Ampere architecture.

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
ASSERT_GE(prop.major, 8) << "Requires Ampere (SM 8.0+) architecture";
```

### Step 2: Allocate Device Memory
Call cudaMalloc with the specified allocation size and capture the returned pointer and error code.

```cpp
void* d_ptr = nullptr;
cudaError_t err = cudaMalloc(&d_ptr, allocation_size);
```

### Step 3: Verify Allocation Success
For valid allocation sizes within available memory, verify:
- Return code is `cudaSuccess`
- Returned pointer is non-null
- Pointer is properly aligned (256-byte boundary minimum)

```cpp
ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
ASSERT_NE(d_ptr, nullptr);
ASSERT_EQ(reinterpret_cast<uintptr_t>(d_ptr) % 256, 0) << "Memory not aligned";
```

### Step 4: Verify Memory Accessibility
Write a test pattern to the allocated memory and read it back to confirm the memory is accessible.

```cpp
cudaMemset(d_ptr, 0xAB, allocation_size);
cudaDeviceSynchronize();
ASSERT_EQ(cudaGetLastError(), cudaSuccess);
```

### Step 5: Deallocate Memory
Free the allocated memory and verify the operation succeeds.

```cpp
err = cudaFree(d_ptr);
ASSERT_EQ(err, cudaSuccess);
```

### Step 6: Verify Error Handling
Attempt allocation exceeding available memory and verify proper error code is returned.

```cpp
void* large_ptr = nullptr;
err = cudaMalloc(&large_ptr, 1ULL << 40);  // 1 TB - should fail
ASSERT_EQ(err, cudaErrorMemoryAllocation);
ASSERT_EQ(large_ptr, nullptr);
```

## Expected Results

| Condition | Expected Outcome |
|-----------|------------------|
| Valid allocation size | cudaSuccess, non-null aligned pointer |
| Zero-size allocation | cudaSuccess, may return null or valid pointer (implementation-defined) |
| Exceeds available memory | cudaErrorMemoryAllocation, null pointer |
| After cudaFree | Pointer invalid, memory returned to pool |

## Error Handling

- If cudaMalloc fails unexpectedly, capture `cudaGetLastError()` and device memory info via `cudaMemGetInfo()`
- Log available vs requested memory for debugging
- Check for memory leaks from previous tests using cuda-memcheck

## Notes

- On multi-GPU systems, ensure correct device is selected before allocation
- Memory fragmentation may cause large allocations to fail even when total free memory is sufficient
- Consider testing with CUDA_LAUNCH_BLOCKING=1 for synchronous error detection

---

# Test Case 2: Advanced Memory Allocation and Data Transfer (CUDA-MEM-006)

## Overview

This test case validates core and advanced memory management functions on Hopper architecture GPUs. It covers the allocation of 2D pitched memory via `cudaMallocPitch` to ensure optimal alignment for 2D array access, managed memory allocation via `cudaMallocManaged`, and compares synchronous (`cudaMemcpy`) versus asynchronous (`cudaMemcpyAsync`) data transfer mechanisms.

## Requirements Reference

- REQ-MEM-004: `cudaMallocPitch` shall provide a pitch value that ensures aligned access for 2D arrays.
- REQ-MEM-005: `cudaMemcpyAsync` shall overlap data transfer with host execution when using non-default streams.
- REQ-MEM-006: `cudaMallocManaged` shall allocate memory accessible by both host and device without manual copies.

## Preconditions

- NVIDIA GPU with Hopper architecture (SM 9.0+) is available.
- CUDA Toolkit 12.2 or later is installed.
- Sufficient device memory for large 2D array allocations.

## Test Data

| Allocation Type | Dimensions / Size | Expected Result | Notes |
|-----------------|-------------------|-----------------|-------|
| 2D Pitched | 4096 x 4096 (float) | Success | Verify `pitch >= width * sizeof(T)` |
| Managed Memory | 1 GB | Success | Single pointer accessibility |
| Async Transfer | 256 MB | Success | Overlap with stream work |

## Test Steps

### Step 1: 2D Pitched Allocation
Allocate a 2D array and verify that the returned pitch meets hardware alignment requirements.

```cpp
float* d_ptr;
size_t pitch;
size_t width = 4096;
size_t height = 4096;

cudaError_t err = cudaMallocPitch(&d_ptr, &pitch, width * sizeof(float), height);
ASSERT_EQ(err, cudaSuccess);
ASSERT_GE(pitch, width * sizeof(float)) << "Pitch cannot be smaller than width";
```

### Step 2: Synchronous 2D Copy
Copy data from host to the pitched device allocation using `cudaMemcpy2D`.

```cpp
float* h_data = (float*)malloc(width * height * sizeof(float));
err = cudaMemcpy2D(d_ptr, pitch, h_data, width * sizeof(float),
                   width * sizeof(float), height, cudaMemcpyHostToDevice);
ASSERT_EQ(err, cudaSuccess);
```

### Step 3: Asynchronous Linear Transfer
Perform a linear memory transfer in a non-default stream to verify non-blocking behavior.

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

float *d_linear, *h_linear;
size_t size = 1024 * 1024 * sizeof(float);
cudaMalloc(&d_linear, size);
h_linear = (float*)malloc(size);

err = cudaMemcpyAsync(d_linear, h_linear, size, cudaMemcpyHostToDevice, stream);
ASSERT_EQ(err, cudaSuccess);
// Host should reach this point before transfer completes
```

### Step 4: Managed Memory Integration
Allocate managed memory and verify it can be used as a source for device-to-device transfers.

```cpp
float* managed_ptr;
cudaMallocManaged(&managed_ptr, size);

// Populate on Host
for(int i=0; i<100; i++) managed_ptr[i] = 1.0f;

// Copy Managed to Device Linear
err = cudaMemcpy(d_linear, managed_ptr, size, cudaMemcpyDeviceToDevice);
ASSERT_EQ(err, cudaSuccess);
```

### Step 5: Cleanup
Release all allocated resources.

```cpp
cudaFree(d_ptr);
cudaFree(d_linear);
cudaFree(managed_ptr);
free(h_data);
free(h_linear);
cudaStreamDestroy(stream);
```

## Expected Results

| Condition | Expected Outcome |
|-----------|------------------|
| Pitched Access | Hardware-aligned pitch returned; memory accessible via `(float*)((char*)base + row * pitch) + col`. |
| Async Execution | `cudaMemcpyAsync` returns control to host immediately. |
| Managed Coherency | Data written on host is visible to the device during `cudaMemcpy`. |

## Error Handling

- If `cudaMallocPitch` fails with `cudaErrorMemoryAllocation`, check for memory fragmentation.
- Ensure `cudaStreamSynchronize` is called after `cudaMemcpyAsync` before accessing data on the host to avoid race conditions.
- Verify that host memory used with `cudaMemcpyAsync` is page-locked (pinned) via `cudaHostAlloc` for true overlap.

## Notes

- `cudaMallocPitch` is highly recommended for 2D data to avoid performance penalties associated with misaligned global memory access.
- On Hopper architecture, asynchronous copies can utilize multiple hardware copy engines for bi-directional transfers.
- Managed memory (`cudaMallocManaged`) behavior may vary based on whether the system supports "Concurrent Managed Access".

---

## Related Test Cases (All Memory Management)

- CUDA-MEM-002: cudaFree validation
- CUDA-MEM-003: cudaMallocManaged unified memory allocation
- CUDA-MEM-004: cudaMallocPitch for 2D allocations
- CUDA-MEM-005: Memory pool (cudaMallocAsync) operations
- CUDA-STRM-001: Stream and Event synchronization
- CUDA-UNIMEM-001: Managed memory prefetching and hints

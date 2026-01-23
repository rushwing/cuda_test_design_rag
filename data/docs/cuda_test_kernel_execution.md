---
gpu_architecture: Hopper
product_series: Data Center
product: H100
module_name: Kernel Execution
cuda_solution_version: "12.2"
cuda_module_version: "12.2.0"
story_id: CUDA-KERN-001
story_description: Verify kernel launch configurations and grid/block dimension limits
priority: High
test_type: Functional
author: CUDA QA Team
created_date: "2024-03-10"
last_updated: "2024-07-15"
tags:
  - kernel
  - launch
  - grid
  - block
  - threads
---

# Test Case: Kernel Launch Configuration Validation

## Overview

This test case validates CUDA kernel launch configurations on Hopper architecture GPUs. It verifies that kernel launches respect hardware limits for grid dimensions, block dimensions, and total thread counts. The test ensures proper error reporting when launch configurations exceed device capabilities.

## Requirements Reference

- REQ-KERN-001: Kernel launch must succeed with valid grid and block dimensions
- REQ-KERN-002: Block size must not exceed 1024 threads (device limit)
- REQ-KERN-003: Grid dimensions must not exceed device maxGridSize limits
- REQ-KERN-004: Invalid launch configurations must return cudaErrorInvalidConfiguration

## Preconditions

- NVIDIA GPU with Hopper architecture (SM 9.0+) is available
- CUDA Toolkit 12.2 or later is installed
- Device is idle with no active kernels
- Sufficient shared memory available for test kernels

## Test Data

| Grid Dim (x,y,z) | Block Dim (x,y,z) | Expected Result | Reason |
|------------------|-------------------|-----------------|--------|
| (1,1,1) | (1,1,1) | Success | Minimum valid config |
| (1,1,1) | (1024,1,1) | Success | Max threads per block |
| (1,1,1) | (32,32,1) | Success | 1024 threads, 2D block |
| (1,1,1) | (1025,1,1) | Fail | Exceeds max threads |
| (65535,1,1) | (256,1,1) | Success | Large grid X |
| (65535,65535,1) | (256,1,1) | Success | Large grid X,Y |
| (2^31-1,1,1) | (256,1,1) | Success | Max grid X dimension |

## Test Steps

### Step 1: Query Device Limits
Retrieve device properties to determine hardware limits for the test GPU.

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

int maxThreadsPerBlock = prop.maxThreadsPerBlock;  // 1024
int maxBlockDimX = prop.maxThreadsDim[0];          // 1024
int maxBlockDimY = prop.maxThreadsDim[1];          // 1024
int maxBlockDimZ = prop.maxThreadsDim[2];          // 64
int maxGridDimX = prop.maxGridSize[0];             // 2^31-1
int maxGridDimY = prop.maxGridSize[1];             // 65535
int maxGridDimZ = prop.maxGridSize[2];             // 65535
```

### Step 2: Define Test Kernel
Create a simple kernel that records execution for validation.

```cpp
__global__ void testKernel(int* output, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        output[0] = value;
    }
}
```

### Step 3: Test Valid Launch Configurations
Launch kernels with valid configurations and verify successful execution.

```cpp
int* d_output;
cudaMalloc(&d_output, sizeof(int));

dim3 grid(1, 1, 1);
dim3 block(256, 1, 1);

testKernel<<<grid, block>>>(d_output, 42);
cudaError_t err = cudaGetLastError();
ASSERT_EQ(err, cudaSuccess) << "Valid launch failed: " << cudaGetErrorString(err);

cudaDeviceSynchronize();
ASSERT_EQ(cudaGetLastError(), cudaSuccess);
```

### Step 4: Test Maximum Block Size
Verify kernel launches at the maximum threads per block limit.

```cpp
dim3 maxBlock(1024, 1, 1);
testKernel<<<1, maxBlock>>>(d_output, 100);
ASSERT_EQ(cudaGetLastError(), cudaSuccess);

dim3 maxBlock2D(32, 32, 1);  // 1024 threads
testKernel<<<1, maxBlock2D>>>(d_output, 101);
ASSERT_EQ(cudaGetLastError(), cudaSuccess);
```

### Step 5: Test Invalid Block Size
Verify proper error when block size exceeds limits.

```cpp
dim3 invalidBlock(1025, 1, 1);
testKernel<<<1, invalidBlock>>>(d_output, 0);
cudaError_t err = cudaGetLastError();
ASSERT_EQ(err, cudaErrorInvalidConfiguration)
    << "Expected cudaErrorInvalidConfiguration, got: " << cudaGetErrorString(err);
```

### Step 6: Test Large Grid Dimensions
Verify kernel launches with large grid dimensions succeed.

```cpp
dim3 largeGrid(65535, 65535, 1);
dim3 smallBlock(32, 1, 1);
testKernel<<<largeGrid, smallBlock>>>(d_output, 200);
ASSERT_EQ(cudaGetLastError(), cudaSuccess);
cudaDeviceSynchronize();
```

### Step 7: Cleanup
Free allocated resources.

```cpp
cudaFree(d_output);
```

## Expected Results

| Test Scenario | Expected Error Code | Expected Behavior |
|---------------|--------------------|--------------------|
| Valid configuration | cudaSuccess | Kernel executes correctly |
| Block > 1024 threads | cudaErrorInvalidConfiguration | Launch rejected, no execution |
| Grid exceeds limits | cudaErrorInvalidConfiguration | Launch rejected, no execution |
| Zero grid or block | cudaErrorInvalidConfiguration | Launch rejected |

## Error Handling

- Capture launch errors immediately after kernel invocation with `cudaGetLastError()`
- Use `cudaDeviceSynchronize()` followed by `cudaGetLastError()` to catch async execution errors
- Log device properties when configuration errors occur for debugging

## Related Test Cases

- CUDA-KERN-002: Shared memory configuration per kernel
- CUDA-KERN-003: Dynamic parallelism (nested kernel launches)
- CUDA-KERN-004: Cooperative kernel launches
- CUDA-KERN-005: Kernel timeout and watchdog behavior

## Notes

- Block dimension limits (1024, 1024, 64) are independent but total threads cannot exceed maxThreadsPerBlock
- Grid dimension X supports up to 2^31-1 on compute capability 3.0+
- Consider testing with different SM occupancy levels
- Hopper introduces Thread Block Clusters; additional tests in CUDA-KERN-010

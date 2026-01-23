---

# GPU and Product Information

gpu_architecture: Hopper
product_series: Data Center
product: H100

# Module and Version Information

module_name: Device Management
cuda_solution_version: "12.2"
cuda_module_version: "12.2.0"

# Story/Ticket Information

story_id: CUDA-DEV-001
story_description: Verify device selection, property retrieval, and state reset operations
priority: High
test_type: Functional

# Metadata

author: CUDA QA Team
created_date: "2024-04-05"
last_updated: "2024-04-05"
tags:

* device-management
* cudaSetDevice
* device-properties
* reset

---

# Test Case: Device Selection and Lifecycle Management

## Overview

This test case validates the fundamental device management operations on Hopper architecture GPUs. It ensures that the system can correctly identify and query H100 hardware capabilities using `cudaGetDeviceProperties`, switch execution contexts between multiple GPUs using `cudaSetDevice`, and perform a clean state teardown using `cudaDeviceReset`. Proper device management is essential for multi-GPU load balancing and error recovery.

## Requirements Reference

* REQ-DEV-001: `cudaGetDeviceProperties` shall return accurate architectural details, including compute capability (9.0 for Hopper).
* REQ-DEV-002: `cudaSetDevice` shall successfully change the active device context for the calling host thread.
* REQ-DEV-003: `cudaDeviceReset` shall explicitly destroy all primary context resources associated with the current device.

## Preconditions

* At least one NVIDIA GPU with Hopper architecture (SM 9.0+) is available.
* CUDA Toolkit 12.2 or later is installed.
* (Optional) Multi-GPU environment for full validation of device switching.

## Test Data

| Function | Parameter | Expected Result | Notes |
| --- | --- | --- | --- |
| cudaGetDeviceProperties | Device ID 0 | Success | Verify `major == 9`, `minor == 0` |
| cudaSetDevice | Valid Device ID | Success | Target device becomes active |
| cudaSetDevice | -1 or Out of Range | cudaErrorInvalidDevice | Boundary test |
| cudaDeviceReset | N/A | Success | All allocations invalidated |

## Test Steps

### Step 1: Query Device Properties

Retrieve and validate the hardware properties of the H100 GPU to ensure the driver recognizes the Hopper architecture correctly.

```cpp
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);
ASSERT_GT(deviceCount, 0);

cudaDeviceProp prop;
cudaError_t err = cudaGetDeviceProperties(&prop, 0);
ASSERT_EQ(err, cudaSuccess);

// Verify Hopper-specific compute capability
ASSERT_EQ(prop.major, 9);
ASSERT_EQ(prop.minor, 0);
printf("Testing on Device: %s\n", prop.name);

```

### Step 2: Set Active Device

In a multi-GPU system, switch the active context to a specific device and verify the selection.

```cpp
int targetDevice = 0; // Or index of another available H100
err = cudaSetDevice(targetDevice);
ASSERT_EQ(err, cudaSuccess);

int currentDevice;
cudaGetDevice(&currentDevice);
ASSERT_EQ(currentDevice, targetDevice);

```

### Step 3: Allocate and Initialize State

Create dummy resources on the device to prepare for the reset test.

```cpp
void* d_ptr;
size_t size = 1024 * 1024;
ASSERT_EQ(cudaMalloc(&d_ptr, size), cudaSuccess);
ASSERT_EQ(cudaMemset(d_ptr, 0, size), cudaSuccess);

```

### Step 4: Perform Device Reset

Reset the device to clear all allocations and state. This simulates an application shutdown or a recovery from a fatal execution error.

```cpp
err = cudaDeviceReset();
ASSERT_EQ(err, cudaSuccess);

```

### Step 5: Verify Post-Reset State

Confirm that previous pointers are now invalid and a new allocation is required.

```cpp
// Subsequent attempts to use d_ptr should fail or require new context initialization
void* d_ptr_new;
err = cudaMalloc(&d_ptr_new, size);
ASSERT_EQ(err, cudaSuccess); // Context is lazily re-initialized
cudaFree(d_ptr_new);

```

## Expected Results

| Condition | Expected Outcome |
| --- | --- |
| Property Check | Returns `9.0` for compute capability and correct VRAM size. |
| Device Switching | `cudaGetDevice` reflects the ID passed to `cudaSetDevice`. |
| Device Reset | All memory, streams, and events are destroyed; returns `cudaSuccess`. |
| Error Scenario | Passing an invalid index to `cudaSetDevice` returns `cudaErrorInvalidDevice`. |

## Error Handling

* If `cudaGetDeviceProperties` returns `cudaErrorInsufficientDriver`, ensure the NVIDIA driver version supports CUDA 12.2.
* `cudaDeviceReset` should only be called when the host thread is finished with the device; any subsequent use of existing handles (streams, pointers) will result in undefined behavior or `cudaErrorContextIsDestroyed`.
* Use `cudaGetLastError()` after `cudaSetDevice` to catch cases where the device is in "Prohibited" compute mode.

## Related Test Cases

* CUDA-MEM-001: Device Memory Allocation.
* CUDA-STRM-001: Stream Operations (Context-bound).
* CUDA-MULTI-001: Peer-to-Peer Device Communication.

## Notes

* For Hopper H100, properties will reflect new features like "Thread Block Clusters" and "Memory Management" capabilities.
* `cudaDeviceReset` is a heavy operation and should not be used for routine cleanup; `cudaFree` and `cudaStreamDestroy` are preferred.
* In MIG (Multi-Instance GPU) mode, `cudaGetDeviceCount` will return the number of accessible MIG instances rather than physical GPUs.

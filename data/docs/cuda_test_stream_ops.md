---

# GPU and Product Information

gpu_architecture: Hopper
product_series: Data Center
product: H100

# Module and Version Information

module_name: Stream Operations
cuda_solution_version: "12.2"
cuda_module_version: "12.2.0"

# Story/Ticket Information

story_id: CUDA-STRM-001
story_description: Verify stream creation, synchronization, and event-based stream waiting
priority: High
test_type: Functional

# Metadata

author: CUDA QA Team
created_date: "2024-03-15"
last_updated: "2024-03-15"
tags:
  - streams
  - synchronization
  - events
  - concurrency

---

# Test Case: Stream Creation and Inter-Stream Synchronization

## Overview

This test case validates the lifecycle and synchronization of CUDA streams on Hopper architecture GPUs. It specifically verifies the creation of non-default streams using `cudaStreamCreate`, host-side synchronization via `cudaStreamSynchronize`, and device-side inter-stream dependencies using `cudaStreamWaitEvent`. These operations are fundamental for achieving overlapping execution of kernels and memory transfers.

## Requirements Reference

* REQ-STRM-001: The system shall support the creation and destruction of multiple independent execution streams.
* REQ-STRM-002: `cudaStreamSynchronize` must block the host until all preceding commands in the specified stream have completed.
* REQ-STRM-003: `cudaStreamWaitEvent` must ensure a stream waits for a specific event to be recorded in another stream before proceeding with subsequent tasks.

## Preconditions

* NVIDIA GPU with Hopper architecture (SM 9.0+) is available.
* CUDA Toolkit 12.2 or later is installed.
* Device supports concurrent kernel execution and multiple hardware work queues.

## Test Data

| Input Parameter | Value | Expected Result | Notes |
| --- | --- | --- | --- |
| Stream Count | 2 | Success | Two independent streams |
| Event Type | cudaEventDefault | Success | Standard event for sync |
| Sync Method | StreamSynchronize | Success | Host-side block |
| Dependency | Stream B waits for A | Success | Orderly execution |

## Test Steps

### Step 1: Initialize Streams and Events

Create two independent streams and one event to facilitate synchronization.

```cpp
cudaStream_t streamA, streamB;
cudaEvent_t event;
 
ASSERT_EQ(cudaStreamCreate(&streamA), cudaSuccess);
ASSERT_EQ(cudaStreamCreate(&streamB), cudaSuccess);
ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);

```

### Step 2: Launch Work and Record Event in Stream A

Launch a kernel in Stream A and record an event immediately following it.

```cpp
// Dummy kernel to simulate workload
testKernel<<<grid, block, 0, streamA>>>(d_outA, valA);
 
// Record event in Stream A
ASSERT_EQ(cudaEventRecord(event, streamA), cudaSuccess);

```

### Step 3: Implement Inter-Stream Dependency

Use `cudaStreamWaitEvent` to make Stream B wait for the event recorded in Stream A. This ensures Stream B does not execute its kernel until Stream A's kernel is finished.

```cpp
// Stream B waits for Stream A's event
ASSERT_EQ(cudaStreamWaitEvent(streamB, event, 0), cudaSuccess);
 
// Launch kernel in Stream B - will wait for event on device side
testKernel<<<grid, block, 0, streamB>>>(d_outB, valB);

```

### Step 4: Synchronize Stream B

Call `cudaStreamSynchronize` on Stream B. This should block the host until both the dependency (Stream A's work) and Stream B's work are complete.

```cpp
ASSERT_EQ(cudaStreamSynchronize(streamB), cudaSuccess);

```

### Step 5: Verify Results and Cleanup

Confirm data integrity and destroy resources.

```cpp
// Verify data from both streams is correct
verifyResults(d_outA, d_outB);
 
ASSERT_EQ(cudaStreamDestroy(streamA), cudaSuccess);
ASSERT_EQ(cudaStreamDestroy(streamB), cudaSuccess);
ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);

```

## Expected Results

| Condition | Expected Outcome |
| --- | --- |
| Stream/Event Creation | `cudaSuccess` for all handles |
| `cudaStreamWaitEvent` | Stream B execution is deferred until Stream A reaches the event |
| `cudaStreamSynchronize` | Host blocks correctly; returns `cudaSuccess` only after completion |
| Resource Cleanup | All handles destroyed without error |

## Error Handling

* If `cudaStreamCreate` fails, verify if the maximum limit of streams or contexts has been reached.
* Use `cudaStreamQuery` to check if a stream is busy without blocking the host during debug.
* Ensure `cudaEventRecord` is called before `cudaStreamWaitEvent` to avoid undefined behavior or deadlocks.

## Related Test Cases

* CUDA-STRM-002: Verify stream priorities and scheduling.
* CUDA-EVNT-001: Comprehensive `cudaEvent` timing and lifecycle.
* CUDA-KERN-001: Kernel launch configuration validation.

## Notes

* `cudaStreamWaitEvent` is a device-side synchronization; the host is not blocked during the wait.
* On Hopper architecture, concurrent streams leverage Hardware Task Descriptions for low-latency scheduling.
* Improper use of the NULL (default) stream can cause unintended serialization across all other streams.


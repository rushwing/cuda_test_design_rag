---

# GPU and Product Information

gpu_architecture: Hopper
product_series: Data Center
product: H100

# Module and Version Information

module_name: Graph Operations
cuda_solution_version: "12.2"
cuda_module_version: "12.2.0"

# Story/Ticket Information

story_id: CUDA-GRAPH-001
story_description: Verify creation, instantiation, and execution of CUDA Graphs for task orchestration
priority: High
test_type: Functional

# Metadata

author: CUDA QA Team
created_date: "2024-04-01"
last_updated: "2024-04-01"
tags:
  - graphs
  - cudaGraphCreate
  - cudaGraphLaunch
  - orchestration

---

# Test Case: CUDA Graph Creation and Execution Lifecycle

## Overview

This test case validates the lifecycle of CUDA Graphs on Hopper architecture GPUs. It verifies the ability to define a workload as a directed acyclic graph (DAG) using `cudaGraphCreate`, instantiate it into an executable graph, and execute it using `cudaGraphLaunch`. CUDA Graphs reduce driver overhead by defining the entire workflow once and launching it repeatedly, which is highly efficient for the massive parallelism of H100 systems.

## Requirements Reference

* REQ-GRAPH-001: The system shall support the creation of a graph container via `cudaGraphCreate`.
* REQ-GRAPH-002: A graph must be successfully instantiated into an executable graph (cudaGraphExec_t).
* REQ-GRAPH-003: `cudaGraphLaunch` must execute all nodes in the graph according to their defined dependencies.
* REQ-GRAPH-004: `cudaGraphExecDestroy` must properly release all resources associated with the executable graph.

## Preconditions

* NVIDIA GPU with Hopper architecture (SM 9.0+) is available.
* CUDA Toolkit 12.2 or later is installed.
* Sufficient device memory is available for the test kernels and graph metadata.

## Test Data

| Node Type | Dependencies | Expected Result | Notes |
| --- | --- | --- | --- |
| Kernel Node A | None | Success | Root node |
| Kernel Node B | Node A | Success | Sequential dependency |
| Kernel Node C | Node A | Success | Parallel execution with B |
| Executable Graph | All Nodes | Success | Reusable execution unit |

## Test Steps

### Step 1: Create Graph and Define Nodes

Initialize a graph and add a kernel node to it.

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Define kernel parameters and add a node
cudaKernelNodeParams nodeParams = {0};
// ... populate nodeParams with testKernel ...
cudaGraphNode_t aNode;
cudaGraphAddKernelNode(&aNode, graph, NULL, 0, &nodeParams);

```

### Step 2: Instantiate the Executable Graph

Compile the graph into an executable format. This stage performs the heavy validation of the graph structure.

```cpp
cudaGraphExec_t graphExec;
cudaError_t err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
ASSERT_EQ(err, cudaSuccess) << "Graph instantiation failed: " << cudaGetErrorString(err);

```

### Step 3: Launch the Graph

Execute the entire graph in a specific stream.

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

err = cudaGraphLaunch(graphExec, stream);
ASSERT_EQ(err, cudaSuccess) << "Graph launch failed: " << cudaGetErrorString(err);

cudaStreamSynchronize(stream);

```

### Step 4: Cleanup Executable Resources

Destroy the executable graph and the graph definition.

```cpp
err = cudaGraphExecDestroy(graphExec);
ASSERT_EQ(err, cudaSuccess);

cudaGraphDestroy(graph);
cudaStreamDestroy(stream);

```

## Expected Results

| Condition | Expected Outcome |
| --- | --- |
| Graph Creation | `cudaSuccess`; valid `cudaGraph_t` handle returned. |
| Instantiation | `cudaSuccess`; driver validates the DAG structure. |
| Launch | All kernels within the graph execute in the correct order. |
| Destruction | `cudaSuccess`; memory is returned to the system without leaks. |

## Error Handling

* If `cudaGraphInstantiate` fails with `cudaErrorInvalidValue`, check for cycles in the node dependencies.
* Use `cudaGraphExecUpdate` if the graph structure remains the same but kernel parameters change to avoid full re-instantiation.
* Capture errors during launch with `cudaGetLastError()` to identify issues within specific graph nodes.

## Related Test Cases

* CUDA-KERN-001: Kernel Launch Configuration Validation.
* CUDA-STRM-001: Stream Operations and Synchronization.
* CUDA-MEM-006: Memory Management and Pitched Allocations.

## Notes

* Hopper GPUs feature optimized hardware scheduling that benefits significantly from the pre-computed nature of CUDA Graphs.
* Graphs can be captured from existing stream-based code using `cudaStreamBeginCapture`, providing an alternative to the explicit API tested here.
* `cudaGraphExecDestroy` does not destroy the `cudaGraph_t` from which it was created; both must be managed independently.

---


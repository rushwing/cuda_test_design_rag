---
# GPU and Product Information
gpu_architecture: <Ampere|Hopper|Ada Lovelace|Turing|Volta>
product_series: <GeForce RTX 30|GeForce RTX 40|Data Center|Quadro|Tesla>
product: <specific product name, e.g., RTX 4090, H100, A100>

# Module and Version Information
module_name: <Memory Management|Kernel Execution|Stream Operations|Synchronization|Error Handling|Device Management|Texture Operations|Graph Operations|Multi-GPU|Unified Memory>
cuda_solution_version: "<major.minor, e.g., 12.0>"
cuda_module_version: "<major.minor.patch, e.g., 12.0.1>"

# Story/Ticket Information
story_id: <CUDA-XXX-NNN format, e.g., CUDA-MEM-001, CUDA-KERN-002>
story_description: <Brief one-line description of the test objective>
priority: <High|Medium|Low>
test_type: <Functional|Performance|Stress|Boundary|Error Handling|Compatibility>

# Metadata
author: CUDA QA Team
created_date: "<YYYY-MM-DD>"
last_updated: "<YYYY-MM-DD>"
tags:
  - <tag1>
  - <tag2>
  - <tag3>
---

# Test Case: <Descriptive Title>

## Overview

<2-3 sentences describing what this test validates and why it matters. Include the specific CUDA functionality being tested and any architecture-specific considerations.>

## Requirements Reference

- REQ-XXX-001: <Requirement description>
- REQ-XXX-002: <Requirement description>
- REQ-XXX-003: <Requirement description>

## Preconditions

- <Hardware requirement>
- <Software/driver requirement>
- <Environment requirement>
- <State requirement>

## Test Data

| Input Parameter | Value | Expected Result | Notes |
|-----------------|-------|-----------------|-------|
| <param1> | <value1> | <result1> | <notes1> |
| <param2> | <value2> | <result2> | <notes2> |

## Test Steps

### Step 1: <Step Title>
<Description of what this step does and why>

```cpp
// Sample code for this step
<code>
```

### Step 2: <Step Title>
<Description of what this step does and why>

```cpp
// Sample code for this step
<code>
```

### Step 3: <Step Title>
<Description of what this step does and why>

```cpp
// Sample code for this step
<code>
```

### Step N: Cleanup
<Description of cleanup/teardown>

```cpp
// Cleanup code
<code>
```

## Expected Results

| Condition | Expected Outcome |
|-----------|------------------|
| <condition1> | <outcome1> |
| <condition2> | <outcome2> |

## Error Handling

- <Error scenario 1 and how to handle/debug>
- <Error scenario 2 and how to handle/debug>

## Related Test Cases

- <STORY-ID-1>: <Brief description>
- <STORY-ID-2>: <Brief description>

## Notes

- <Important consideration or caveat>
- <Architecture-specific behavior>
- <Known issues or limitations>

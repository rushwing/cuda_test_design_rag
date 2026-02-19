"""Prompt templates for CUDA test generation."""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class TestPromptTemplates:
    """Collection of prompt templates for test generation."""

    SYSTEM_TEMPLATE = """You are an expert CUDA test engineer. Your task is to generate
comprehensive test suites for CUDA kernels and GPU computing code based on requirements
and design documents.

When generating tests, consider:
- Memory allocation and deallocation (cudaMalloc, cudaFree)
- Kernel launch configurations (grid/block dimensions)
- Data transfer between host and device (cudaMemcpy)
- Synchronization (cudaDeviceSynchronize)
- Error handling (cudaGetLastError)
- Edge cases and boundary conditions
- Performance considerations
- Different GPU architectures compatibility

Use the provided context from requirements and design documents to generate accurate
and relevant test cases."""

    TEST_GENERATION_TEMPLATE = """Based on the following context from requirements and
design documents, generate a comprehensive CUDA test suite.

Context:
{context}

User Request:
{query}

Generate test code that:
1. Includes necessary CUDA headers and test framework imports
2. Has clear test case names and descriptions
3. Tests both success and failure scenarios
4. Includes memory management tests
5. Validates kernel correctness
6. Handles edge cases

Provide the test code with detailed comments explaining each test case."""

    # ========== Stage 1: Test Intent Generation ==========

    INTENT_SYSTEM_TEMPLATE = """You are an expert CUDA test architect. Your task is to analyze
requirements and design documents to extract comprehensive test intents.

A test intent describes WHAT should be tested and WHY, without specifying HOW (implementation).
Each test intent should be:
- Specific and actionable
- Traceable to a requirement or design specification
- Independent and self-contained
- Focused on a single testing concern

Categories of CUDA test intents to consider:
- Functional correctness (kernel logic, algorithms)
- Memory management (allocation, transfer, boundaries)
- Error handling (invalid inputs, resource exhaustion)
- Performance (throughput, latency, occupancy)
- Synchronization (streams, events, barriers)
- Edge cases (empty inputs, maximum sizes, boundary values)
- Compatibility (different GPU architectures, compute capabilities)"""

    INTENT_GENERATION_TEMPLATE = """Based on the following context from requirements and
design documents, extract comprehensive test intents.

Context:
{context}

User Request:
{query}

For each test intent, provide:
1. **Intent ID**: A unique identifier (e.g., TI-001)
2. **Category**: The testing category (Functional/Memory/Error/Performance/Sync/Edge/Compat)
3. **Title**: A concise title describing what to test
4. **Description**: Detailed description of the test intent
5. **Requirement Reference**: Which requirement/spec this traces to (if identifiable)
6. **Priority**: High/Medium/Low based on criticality
7. **Preconditions**: Required setup or conditions

Output the test intents in the following YAML format:

```yaml
test_intents:
  - id: TI-001
    category: Functional
    title: "Brief title"
    description: "Detailed description of what should be tested and why"
    requirement_ref: "REQ-XXX or section reference"
    priority: High
    preconditions:
      - "Precondition 1"
      - "Precondition 2"
```

Generate all relevant test intents based on the provided context."""

    # ========== Stage 2: Test Case Generation (Markdown) ==========

    TESTCASE_SYSTEM_TEMPLATE = """You are an expert CUDA test engineer. Your task is to
generate detailed test case documents based on provided test intents.

Each test case should be a complete, human-readable document that includes:
- Clear test objective and scope
- Detailed preconditions and setup requirements
- Step-by-step test procedure
- Expected results and pass/fail criteria
- Test data with boundary values and edge cases
- Error scenarios to verify

Follow the exact markdown format with YAML front matter as shown in the examples."""

    TESTCASE_GENERATION_TEMPLATE = """Based on the following test intents, generate
detailed test case documents in markdown format.

## Test Intents to Expand:
{test_intents}

## Reference Context (from knowledge base):
{context}

## Output Requirements:
Generate ONE markdown document containing all test cases. Each test case should follow this format:

```markdown
---
gpu_architecture: {gpu_architecture}
product_series: {product_series}
module_name: {module_name}
cuda_solution_version: "{cuda_version}"
story_id: [Generated ID continuing from existing]
story_description: [From intent description]
priority: [From intent]
test_type: Functional
tags:
  - [relevant tags]
---

# Test Case: [Title from Intent]

## Overview
[Expand the intent description into 2-3 sentences explaining what this test validates and why]

## Requirements Reference
[Map to requirement refs from intent]

## Preconditions
[Expand preconditions from intent into detailed list]

## Test Data
| Input | Value | Expected Result | Notes |
|-------|-------|-----------------|-------|
[Include boundary values, edge cases, normal cases]

## Test Steps
### Step 1: [Setup]
[Detailed setup instructions]

### Step 2: [Execute]
[What to execute and how]

### Step 3: [Verify]
[What to check and expected outcomes]

### Step N: [Cleanup]
[Cleanup instructions]

## Expected Results
| Condition | Expected Outcome |
|-----------|------------------|
[Clear pass/fail criteria]

## Error Handling
[Error scenarios and how to handle them]

## Notes
[Any additional considerations]
```

Generate complete test cases for ALL provided intents."""

    # ========== Legacy: Test Skeleton Generation (C++ Code) ==========

    SKELETON_SYSTEM_TEMPLATE = """You are an expert CUDA test implementer. Your task is to
generate test case skeletons based on provided test intents.

A test skeleton should include:
- Complete function signature with proper naming
- Structured sections: Setup, Execute, Verify, Cleanup
- Placeholder comments for implementation details
- Proper CUDA error checking patterns
- Memory management scaffolding
- Assertions structure (without specific values)

Follow CUDA testing best practices:
- Use descriptive test names matching the intent
- Include device capability checks where needed
- Structure for both host and device validation
- Support for parameterized testing where applicable"""

    SKELETON_GENERATION_TEMPLATE = """Based on the following test intents, generate
test case skeletons in CUDA/C++.

Test Intents:
{test_intents}

Additional Context (optional):
{context}

For each test intent, generate a test skeleton that includes:
1. Test function with descriptive name
2. Setup section with TODOs for initialization
3. Execution section with kernel launch placeholder
4. Verification section with assertion structure
5. Cleanup section for resource deallocation
6. Error handling scaffolding

Use GoogleTest (gtest) framework style. Output format:

```cpp
// Test Intent: [ID] - [Title]
// Category: [Category]
// Priority: [Priority]
// Requirement: [Reference]

TEST(TestSuiteName, TestName) {{
    // === SETUP ===
    // TODO: Initialize test data
    // TODO: Allocate device memory

    // === EXECUTE ===
    // TODO: Launch kernel
    // TODO: Synchronize

    // === VERIFY ===
    // TODO: Copy results back
    // TODO: Assert expected outcomes

    // === CLEANUP ===
    // TODO: Free device memory
}}
```

Generate skeletons for all provided test intents."""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt."""
        return cls.SYSTEM_TEMPLATE

    @classmethod
    def get_test_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the test generation prompt template (legacy single-stage)."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.SYSTEM_TEMPLATE),
            ("human", cls.TEST_GENERATION_TEMPLATE),
        ])

    @classmethod
    def get_simple_prompt(cls) -> PromptTemplate:
        """Get a simple prompt template for basic queries."""
        return PromptTemplate(
            input_variables=["context", "query"],
            template=cls.TEST_GENERATION_TEMPLATE,
        )

    # ========== Few-Shot Corner Case Generation ==========

    FEWSHOT_SYSTEM_TEMPLATE = """You are an expert CUDA test engineer with deep knowledge of GPU
architecture, CUDA runtime, and testing best practices.

Your task is to generate NEW test cases that are NOT in the provided examples. Use the examples
as a PATTERN for format and style, but generate NOVEL corner cases and edge cases.

You should leverage your expertise in:
- CUDA memory model edge cases (alignment, coalescing, bank conflicts)
- Kernel launch boundary conditions (max threads, shared memory limits)
- Race conditions and synchronization pitfalls
- Multi-GPU scenarios
- Unified memory page faulting behavior
- Asynchronous execution hazards
- Error propagation across streams
- Resource exhaustion scenarios

DO NOT simply rephrase or slightly modify the examples. Generate genuinely new test scenarios."""

    FEWSHOT_GENERATION_TEMPLATE = """## Task
Generate {num_new_cases} NEW test cases for the specified module. Use the example test cases
below as a PATTERN for format and structure, but create NOVEL corner cases and edge cases.

## Target Configuration
- GPU Architecture: {gpu_architecture}
- Module: {module_name}
- CUDA Version: {cuda_version}
- Focus Areas: {focus_areas}
- Priority: {priority}

## Example Test Cases (Few-Shot - DO NOT COPY, use as format reference only)
{few_shot_examples}

## Existing Test Case IDs to AVOID duplicating
{existing_ids}

## Additional Requirements
{additional_requirements}

## Instructions
1. Study the FORMAT and STRUCTURE of the examples above
2. Generate {num_new_cases} completely NEW test cases that:
   - Cover corner cases NOT in the examples
   - Test edge conditions, race conditions, resource limits
   - Include failure scenarios and error handling
   - Are specific to {gpu_architecture} architecture where relevant
3. Use the same markdown format with YAML front matter
4. Assign new sequential IDs (continue from existing)
5. Include rationale for WHY each test case is important

## Output
Generate the test cases in the same markdown format as the examples."""

    # ========== Multi-Stage Prompt Getters ==========

    @classmethod
    def get_intent_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the Stage 1 test intent generation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.INTENT_SYSTEM_TEMPLATE),
            ("human", cls.INTENT_GENERATION_TEMPLATE),
        ])

    @classmethod
    def get_skeleton_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the Stage 2 test skeleton generation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.SKELETON_SYSTEM_TEMPLATE),
            ("human", cls.SKELETON_GENERATION_TEMPLATE),
        ])

    @classmethod
    def get_fewshot_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the few-shot corner case generation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.FEWSHOT_SYSTEM_TEMPLATE),
            ("human", cls.FEWSHOT_GENERATION_TEMPLATE),
        ])

    @classmethod
    def get_testcase_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the Stage 2 test case generation prompt (markdown format)."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.TESTCASE_SYSTEM_TEMPLATE),
            ("human", cls.TESTCASE_GENERATION_TEMPLATE),
        ])

    # ========== Stage 3: Full C++ Code Generation ==========

    CODE_GEN_SYSTEM_TEMPLATE = """You are an expert CUDA test engineer with deep knowledge of:
- GoogleTest framework
- CUDA runtime APIs (driver and runtime)
- GPU memory management
- Kernel execution patterns
- Error handling best practices
- Performance testing

Your task is to generate COMPLETE, COMPILABLE C++ test code based on test case specifications.
The code must:
- Include all necessary headers
- Have proper namespace usage
- Follow GoogleTest conventions
- Include proper CUDA error checking macros
- Be ready to compile and run"""

    CODE_GEN_TEMPLATE = """Based on the following test case specifications, generate COMPLETE
and COMPILABLE CUDA test code using GoogleTest framework.

## Test Cases:
{test_cases}

## CUDA Context:
{context}

## Requirements:
1. Generate complete, compilable C++ code (not skeletons)
2. Include proper #include statements:
   - <gtest/gtest.h>
   - <cuda_runtime.h>
   - <stdio.h>, <stdlib.h> as needed
3. Define CUDA_CHECK() macro for error checking
4. Create fixture class if shared setup is needed
5. Implement actual kernel launch and verification logic
6. Include proper memory allocation/deallocation
7. Add ASSERT_* and EXPECT_* macros appropriately
8. Include cleanup in TearDown if needed

## Output Format:
```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            FAIL() << "CUDA error"; \\
        } \\
    } while (0)

// Helper functions if needed
static void setupDevice(int deviceId = 0) {{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceId >= deviceCount) deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
}}

// Test fixture (if applicable)
class TestFixtureName : public ::testing::Test {{
protected:
    void SetUp() override {{
        setupDevice();
        // TODO: Allocate test memory
    }}
    
    void TearDown() override {{
        // TODO: Free test memory
        cudaDeviceReset();
    }}
}};

// Actual test implementations
TEST_F(TestFixtureName, TestCaseName) {{
    // Actual implementation with working code
    float* d_data;
    size_t size = N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Kernel launch
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    kernelName<<<grid, block>>>(d_data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verification
    std::vector<float> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {{
        EXPECT_FLOAT_EQ(h_result[i], expected[i]);
    }}
    
    CUDA_CHECK(cudaFree(d_data));
}}

int main(int argc, char** argv) {{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}}
```

Generate complete code for ALL provided test cases. Include realistic test data and assertions."""

    @classmethod
    def get_code_generation_prompt(cls) -> ChatPromptTemplate:
        """Get the Stage 3 full C++ code generation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", cls.CODE_GEN_SYSTEM_TEMPLATE),
            ("human", cls.CODE_GEN_TEMPLATE),
        ])

"""Data models for test generation pipeline."""

import re
from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class TestCategory(str, Enum):
    """Categories of CUDA tests."""

    FUNCTIONAL = "Functional"
    MEMORY = "Memory"
    ERROR = "Error"
    PERFORMANCE = "Performance"
    SYNC = "Sync"
    EDGE = "Edge"
    COMPAT = "Compat"


class TestPriority(str, Enum):
    """Priority levels for test intents."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TestIntent(BaseModel):
    """A single test intent extracted from requirements."""

    id: str = Field(..., description="Unique identifier (e.g., TI-001)")
    category: str = Field(..., description="Testing category")
    title: str = Field(..., description="Concise title")
    description: str = Field(..., description="Detailed description")
    requirement_ref: str = Field(default="", description="Requirement reference")
    priority: str = Field(default="Medium", description="Priority level")
    preconditions: list[str] = Field(default_factory=list, description="Required preconditions")

    def to_summary(self) -> str:
        """Convert to a summary string for LLM consumption."""
        return f"""- {self.id}: {self.title}
  Category: {self.category} | Priority: {self.priority}
  Description: {self.description}
  Preconditions: {', '.join(self.preconditions) if self.preconditions else 'None'}"""


class TestIntentCollection(BaseModel):
    """Collection of test intents from Stage 1."""

    test_intents: list[TestIntent] = Field(default_factory=list)
    source_query: str = Field(default="", description="Original query that generated these intents")
    raw_response: str = Field(default="", description="Raw LLM response for debugging")

    @classmethod
    def from_yaml_string(cls, yaml_str: str, source_query: str = "") -> "TestIntentCollection":
        """Parse test intents from YAML string in LLM response."""
        # Extract YAML block from markdown code fence if present
        yaml_match = re.search(r"```ya?ml\s*(.*?)```", yaml_str, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
        else:
            # Try to find YAML-like content directly
            yaml_content = yaml_str.strip()

        try:
            data = yaml.safe_load(yaml_content)
            if data is None:
                return cls(test_intents=[], source_query=source_query, raw_response=yaml_str)

            intents = []
            intent_list = data.get("test_intents", [])
            if isinstance(intent_list, list):
                for item in intent_list:
                    if isinstance(item, dict):
                        intents.append(TestIntent(
                            id=item.get("id", "TI-???"),
                            category=item.get("category", "Functional"),
                            title=item.get("title", "Untitled"),
                            description=item.get("description", ""),
                            requirement_ref=item.get("requirement_ref", ""),
                            priority=item.get("priority", "Medium"),
                            preconditions=item.get("preconditions", []) or [],
                        ))

            return cls(
                test_intents=intents,
                source_query=source_query,
                raw_response=yaml_str,
            )
        except yaml.YAMLError:
            # Return empty collection with raw response for debugging
            return cls(test_intents=[], source_query=source_query, raw_response=yaml_str)

    def to_prompt_string(self) -> str:
        """Convert to a string suitable for Stage 2 prompt."""
        if not self.test_intents:
            return "No test intents available."

        parts = []
        for intent in self.test_intents:
            parts.append(intent.to_summary())
        return "\n\n".join(parts)

    def to_yaml_string(self) -> str:
        """Serialize test intents to YAML format."""
        data = {
            "test_intents": [
                {
                    "id": intent.id,
                    "category": intent.category,
                    "title": intent.title,
                    "description": intent.description,
                    "requirement_ref": intent.requirement_ref,
                    "priority": intent.priority,
                    "preconditions": intent.preconditions,
                }
                for intent in self.test_intents
            ]
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def filter_by_priority(self, priority: str) -> "TestIntentCollection":
        """Filter intents by priority level."""
        filtered = [i for i in self.test_intents if i.priority.lower() == priority.lower()]
        return TestIntentCollection(
            test_intents=filtered,
            source_query=self.source_query,
            raw_response=self.raw_response,
        )

    def filter_by_category(self, category: str) -> "TestIntentCollection":
        """Filter intents by category."""
        filtered = [i for i in self.test_intents if i.category.lower() == category.lower()]
        return TestIntentCollection(
            test_intents=filtered,
            source_query=self.source_query,
            raw_response=self.raw_response,
        )


class TestSkeleton(BaseModel):
    """A generated test skeleton."""

    intent_id: str = Field(..., description="Reference to the test intent")
    code: str = Field(..., description="Generated test skeleton code")


class TestCase(BaseModel):
    """A generated test case from Stage 2."""

    id: str = Field(..., description="Test case ID")
    title: str = Field(..., description="Test case title")
    description: str = Field(..., description="Detailed test description")
    preconditions: list[str] = Field(default_factory=list, description="Test preconditions")
    test_steps: list[str] = Field(default_factory=list, description="Step-by-step test procedure")
    expected_results: list[str] = Field(default_factory=list, description="Expected outcomes")
    priority: str = Field(default="Medium", description="Priority level")
    category: str = Field(default="Functional", description="Test category")


class TestCaseCollection(BaseModel):
    """Collection of test cases from Stage 2."""

    test_cases: list[TestCase] = Field(default_factory=list)
    source_intents: str = Field(default="", description="Source test intents")
    raw_response: str = Field(default="", description="Raw LLM response for debugging")


class PipelineResult(BaseModel):
    """Complete result from the multi-stage pipeline."""

    query: str = Field(..., description="Original user query")
    retrieved_documents: list[str] = Field(default_factory=list, description="Retrieved doc sources")
    context: str = Field(default="", description="Retrieved context")
    test_intents: TestIntentCollection = Field(
        default_factory=TestIntentCollection,
        description="Generated test intents from Stage 1",
    )
    test_cases: TestCaseCollection = Field(
        default_factory=TestCaseCollection,
        description="Generated test cases from Stage 2",
    )
    test_code: str = Field(default="", description="Generated C++ code from Stage 3")
    stage_completed: int = Field(default=0, description="Last completed stage (1, 2, or 3)")

    def get_summary(self) -> str:
        """Get a summary of the pipeline result."""
        return f"""Pipeline Result Summary:
Query: {self.query}
Documents Retrieved: {len(self.retrieved_documents)}
Test Intents Generated: {len(self.test_intents.test_intents)}
Test Cases Generated: {len(self.test_cases.test_cases)}
Stage Completed: {self.stage_completed}
Code Generated: {'Yes' if self.test_code else 'No'}"""

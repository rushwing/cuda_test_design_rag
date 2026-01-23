"""Data models for request tracking and review workflow."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RequestStatus(str, Enum):
    """Status of a test generation request."""

    # Initial state
    PENDING = "pending"  # Request created, not yet processed

    # Stage 1: Intent generation
    INTENT_GENERATING = "intent_generating"  # Currently generating intents
    INTENT_GENERATED = "intent_generated"  # Intents generated, awaiting review
    INTENT_UNDER_REVIEW = "intent_under_review"  # Expert is reviewing intents
    INTENT_APPROVED = "intent_approved"  # Expert approved intents
    INTENT_REJECTED = "intent_rejected"  # Expert rejected intents
    INTENT_REVISION_REQUESTED = "intent_revision_requested"  # Intents need rework

    # Stage 2: Test case generation
    CASES_GENERATING = "cases_generating"  # Currently generating test cases
    CASES_GENERATED = "cases_generated"  # Test cases generated, awaiting review
    CASES_UNDER_REVIEW = "cases_under_review"  # Expert reviewing test cases
    CASES_APPROVED = "cases_approved"  # Expert approved test cases
    CASES_REJECTED = "cases_rejected"  # Expert rejected test cases

    # Final states
    COMPLETED = "completed"  # Fully complete
    FAILED = "failed"  # Generation failed
    TRAINING_READY = "training_ready"  # Approved and ready for training data


class TestFilters(BaseModel):
    """Structured filters for test generation request."""

    gpu_architecture: str = Field(..., description="Target GPU architecture")
    product_series: Optional[str] = None
    product: Optional[str] = None
    module_name: str = Field(..., description="CUDA module to test")
    cuda_solution_version: str = Field(..., description="CUDA version")
    cuda_module_version: Optional[str] = None
    priority: str = Field(default="Medium")
    test_type: str = Field(default="Functional")


class GenerationConfig(BaseModel):
    """Configuration for test case generation."""

    num_few_shot_examples: int = Field(default=3, ge=1, le=10)
    num_new_cases_to_generate: int = Field(default=5, ge=1, le=20)
    focus_areas: list[str] = Field(default_factory=list)
    exclude_existing_story_ids: list[str] = Field(default_factory=list)
    additional_requirements: str = Field(default="")
    target_framework: str = Field(default="GoogleTest")


class GenerationRequest(BaseModel):
    """A test generation request record."""

    # Identifiers
    request_id: str = Field(..., description="Unique request ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Requester info
    tester_uid: str = Field(..., description="User ID of requester")
    tester_role: str = Field(default="SDET")

    # Request configuration
    filters: TestFilters
    generation_config: GenerationConfig

    # Status tracking
    status: RequestStatus = Field(default=RequestStatus.PENDING)

    # File paths (relative to data directory)
    few_shot_source_paths: list[str] = Field(
        default_factory=list, description="Paths to source test case files used as few-shot"
    )
    generated_intents_path: Optional[str] = Field(
        None, description="Path to generated intents YAML"
    )
    generated_tests_path: Optional[str] = Field(
        None, description="Path to generated test cases markdown"
    )

    # Generation metadata
    llm_model_used: Optional[str] = None
    generation_duration_seconds: Optional[float] = None
    num_cases_generated: Optional[int] = None
    error_message: Optional[str] = None


class ReviewRecord(BaseModel):
    """A review record for generated test cases."""

    review_id: str = Field(..., description="Unique review ID")
    request_id: str = Field(..., description="Associated request ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Reviewer info
    reviewer_uid: str = Field(..., description="User ID of reviewer")
    reviewer_role: str = Field(default="Test Expert")

    # Review outcome
    status: RequestStatus = Field(..., description="Review decision")
    review_notes: str = Field(default="")

    # Quality scores (optional)
    accuracy_score: Optional[int] = Field(None, ge=1, le=5)
    completeness_score: Optional[int] = Field(None, ge=1, le=5)
    relevance_score: Optional[int] = Field(None, ge=1, le=5)

    # File paths for reviewed/modified versions
    reviewed_tests_path: Optional[str] = Field(
        None, description="Path to expert-modified test cases"
    )

    # Training data flag
    approved_for_training: bool = Field(
        default=False, description="Whether approved as training data"
    )

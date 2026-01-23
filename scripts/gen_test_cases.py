#!/usr/bin/env python3
"""Test case generation script with two-stage workflow.

Usage:
    # Auto mode (default): Skip expert review, run both stages
    python scripts/gen_test_cases.py --config config/test_request_sample.json --auto

    # Manual mode - Stage 1 only: Generate intents for review
    python scripts/gen_test_cases.py --config config/test_request_sample.json --intent

    # Manual mode - Stage 2 only: Generate cases from approved intents
    python scripts/gen_test_cases.py --request-id REQ-2024-0123-001 --case

    # List pending requests
    python scripts/gen_test_cases.py --list --status intent_generated
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cuda_test_rag.config import Settings
from cuda_test_rag.core.retriever import DocumentRetriever
from cuda_test_rag.storage import (
    GenerationRequest,
    RequestDatabase,
    RequestStatus,
    TestFileManager,
)
from cuda_test_rag.storage.models import GenerationConfig, TestFilters
from cuda_test_rag.test_generation.models import TestIntentCollection
from cuda_test_rag.test_generation.pipeline import TestGenerationPipeline
from cuda_test_rag.test_generation.prompts import TestPromptTemplates


def generate_request_id() -> str:
    """Generate a unique request ID."""
    now = datetime.utcnow()
    short_uuid = uuid4().hex[:6].upper()
    return f"REQ-{now.strftime('%Y%m%d')}-{short_uuid}"


def load_config(config_path: str) -> dict:
    """Load request configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def run_stage1_intent_generation(
    request: GenerationRequest,
    settings: Settings,
    db: RequestDatabase,
    file_manager: TestFileManager,
) -> bool:
    """Stage 1: Generate test intents from knowledge base.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: Generating Test Intents")
    print(f"Request ID: {request.request_id}")
    print(f"Module: {request.filters.module_name}")
    print(f"GPU Architecture: {request.filters.gpu_architecture}")
    print(f"{'='*60}\n")

    # Update status
    db.update_request_status(request.request_id, RequestStatus.INTENT_GENERATING)

    try:
        start_time = time.time()

        # Initialize pipeline
        pipeline = TestGenerationPipeline(settings)

        # Build query from filters
        query = f"""Generate test intents for {request.filters.module_name} module
on {request.filters.gpu_architecture} architecture with CUDA {request.filters.cuda_solution_version}.
Priority: {request.filters.priority}
Test Type: {request.filters.test_type}

Additional requirements: {request.generation_config.additional_requirements}

Focus areas: {', '.join(request.generation_config.focus_areas) if request.generation_config.focus_areas else 'General coverage'}
"""

        print(f"Query: {query[:200]}...")
        print(f"\nRetrieving relevant documents (k={request.generation_config.num_few_shot_examples})...")

        # Generate intents
        intents, context, doc_sources = pipeline.generate_intents(
            query, k=request.generation_config.num_few_shot_examples
        )

        duration = time.time() - start_time

        print(f"\nRetrieved {len(doc_sources)} documents:")
        for src in doc_sources:
            print(f"  - {src}")

        print(f"\nGenerated {len(intents.test_intents)} test intents:")
        for intent in intents.test_intents:
            print(f"  - {intent.id}: {intent.title} [{intent.priority}]")

        # Save files
        intents_path = file_manager.save_generated_intents(
            request.request_id,
            intents.to_yaml_string(),
            request.created_at,
        )
        print(f"\nIntents saved to: {intents_path}")

        # Save few-shot sources
        file_manager.save_few_shot_sources(
            request.request_id, doc_sources, request.created_at
        )

        # Update database
        db.update_request_status(
            request.request_id,
            RequestStatus.INTENT_GENERATED,
            generated_intents_path=str(intents_path),
            generation_duration_seconds=duration,
            num_cases_generated=len(intents.test_intents),
        )

        print(f"\nStage 1 completed in {duration:.2f}s")
        print(f"Status: INTENT_GENERATED")
        return True

    except Exception as e:
        db.update_request_status(
            request.request_id,
            RequestStatus.FAILED,
            error_message=str(e),
        )
        print(f"\nERROR: {e}")
        return False


def run_stage2_case_generation(
    request: GenerationRequest,
    settings: Settings,
    db: RequestDatabase,
    file_manager: TestFileManager,
) -> bool:
    """Stage 2: Generate test cases from approved intents.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: Generating Test Cases")
    print(f"Request ID: {request.request_id}")
    print(f"{'='*60}\n")

    # Check status
    if request.status not in [
        RequestStatus.INTENT_GENERATED,
        RequestStatus.INTENT_APPROVED,
    ]:
        print(f"ERROR: Cannot generate cases. Current status: {request.status}")
        print("Intents must be in INTENT_GENERATED or INTENT_APPROVED status.")
        return False

    if not request.generated_intents_path:
        print("ERROR: No intents file found for this request.")
        return False

    # Update status
    db.update_request_status(request.request_id, RequestStatus.CASES_GENERATING)

    try:
        start_time = time.time()

        # Load intents
        intents_path = Path(request.generated_intents_path)
        if not intents_path.exists():
            # Try relative to data dir
            intents_path = file_manager.base_path / request.generated_intents_path

        print(f"Loading intents from: {intents_path}")
        intents_content = intents_path.read_text()
        intents = TestIntentCollection.from_yaml_string(f"```yaml\n{intents_content}\n```")

        print(f"Loaded {len(intents.test_intents)} intents")

        # Initialize LLM
        from langchain_core.output_parsers import StrOutputParser
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.temperature,
        )

        # Get prompt template
        prompt = TestPromptTemplates.get_testcase_generation_prompt()

        # Retrieve context for reference
        retriever = DocumentRetriever(settings)
        docs = retriever.retrieve(
            f"{request.filters.module_name} test cases examples",
            k=request.generation_config.num_few_shot_examples,
        )
        context = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

        # Build chain
        chain = prompt | llm | StrOutputParser()

        print("Generating test cases...")

        # Generate test cases
        test_cases = chain.invoke({
            "test_intents": intents.to_prompt_string(),
            "context": context,
            "gpu_architecture": request.filters.gpu_architecture,
            "product_series": request.filters.product_series or "General",
            "module_name": request.filters.module_name,
            "cuda_version": request.filters.cuda_solution_version,
        })

        duration = time.time() - start_time

        # Save generated test cases
        tests_path = file_manager.save_generated_tests(
            request.request_id, test_cases, request.created_at
        )
        print(f"\nTest cases saved to: {tests_path}")

        # Update database
        db.update_request_status(
            request.request_id,
            RequestStatus.CASES_GENERATED,
            generated_tests_path=str(tests_path),
            generation_duration_seconds=duration,
        )

        print(f"\nStage 2 completed in {duration:.2f}s")
        print(f"Status: CASES_GENERATED")
        return True

    except Exception as e:
        db.update_request_status(
            request.request_id,
            RequestStatus.FAILED,
            error_message=str(e),
        )
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def approve_intents(request_id: str, db: RequestDatabase) -> bool:
    """Mark intents as approved (for manual workflow)."""
    request = db.get_request(request_id)
    if not request:
        print(f"Request not found: {request_id}")
        return False

    if request.status != RequestStatus.INTENT_GENERATED:
        print(f"Cannot approve. Current status: {request.status}")
        return False

    db.update_request_status(request_id, RequestStatus.INTENT_APPROVED)
    print(f"Intents approved for request: {request_id}")
    return True


def list_requests(db: RequestDatabase, status: str | None = None):
    """List requests with optional status filter."""
    status_enum = RequestStatus(status) if status else None
    requests = db.list_requests(status=status_enum, limit=20)

    if not requests:
        print("No requests found.")
        return

    print(f"\n{'ID':<25} {'Status':<20} {'Module':<20} {'Created':<20}")
    print("-" * 85)
    for req in requests:
        print(
            f"{req.request_id:<25} {req.status.value:<20} "
            f"{req.filters.module_name:<20} {req.created_at.strftime('%Y-%m-%d %H:%M'):<20}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Test Case Generation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode (both stages, no review)
  python scripts/gen_test_cases.py --config config/test_request_sample.json --auto

  # Stage 1 only (generate intents for review)
  python scripts/gen_test_cases.py --config config/test_request_sample.json --intent

  # Approve intents after review
  python scripts/gen_test_cases.py --approve REQ-20240123-ABC123

  # Stage 2 only (generate cases from approved intents)
  python scripts/gen_test_cases.py --request-id REQ-20240123-ABC123 --case

  # List requests by status
  python scripts/gen_test_cases.py --list --status intent_generated
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: run both stages without expert review (default)",
    )
    mode_group.add_argument(
        "--intent",
        action="store_true",
        help="Stage 1 only: generate test intents",
    )
    mode_group.add_argument(
        "--case",
        action="store_true",
        help="Stage 2 only: generate test cases from intents",
    )
    mode_group.add_argument(
        "--approve",
        metavar="REQUEST_ID",
        help="Approve intents for a request",
    )
    mode_group.add_argument(
        "--list",
        action="store_true",
        help="List requests",
    )

    # Input options
    parser.add_argument(
        "--config",
        "-c",
        help="Path to request configuration JSON file",
    )
    parser.add_argument(
        "--request-id",
        "-r",
        help="Request ID for Stage 2 or status operations",
    )
    parser.add_argument(
        "--status",
        "-s",
        help="Filter by status (for --list)",
    )

    args = parser.parse_args()

    # Initialize
    settings = Settings()
    db = RequestDatabase(settings)
    file_manager = TestFileManager(settings)

    # Handle list
    if args.list:
        list_requests(db, args.status)
        return

    # Handle approve
    if args.approve:
        approve_intents(args.approve, db)
        return

    # Handle Stage 2 only
    if args.case:
        if not args.request_id:
            print("ERROR: --request-id required for --case mode")
            sys.exit(1)

        request = db.get_request(args.request_id)
        if not request:
            print(f"Request not found: {args.request_id}")
            sys.exit(1)

        success = run_stage2_case_generation(request, settings, db, file_manager)
        sys.exit(0 if success else 1)

    # For intent or auto mode, need config
    if not args.config:
        print("ERROR: --config required for --intent or --auto mode")
        sys.exit(1)

    # Load config and create request
    config = load_config(args.config)
    request_id = generate_request_id()

    request = GenerationRequest(
        request_id=request_id,
        tester_uid=config.get("request_metadata", {}).get("tester_uid", "anonymous"),
        tester_role=config.get("request_metadata", {}).get("tester_role", "SDET"),
        filters=TestFilters(**config.get("test_filters", {})),
        generation_config=GenerationConfig(**config.get("generation_config", {})),
        status=RequestStatus.PENDING,
    )

    # Save request config
    file_manager.save_request_config(request_id, config, request.created_at)

    # Create request in database
    db.create_request(request)
    print(f"Created request: {request_id}")

    # Run Stage 1
    if not run_stage1_intent_generation(request, settings, db, file_manager):
        sys.exit(1)

    # If intent-only mode, stop here
    if args.intent:
        print(f"\n{'='*60}")
        print("INTENT MODE: Stage 1 complete.")
        print(f"Review intents at: {request.generated_intents_path or 'See database'}")
        print(f"To approve: python scripts/gen_test_cases.py --approve {request_id}")
        print(f"To generate cases: python scripts/gen_test_cases.py --request-id {request_id} --case")
        print(f"{'='*60}")
        return

    # Auto mode: continue to Stage 2
    print("\n[AUTO MODE] Skipping expert review, proceeding to Stage 2...")

    # Refresh request from database
    request = db.get_request(request_id)

    if not run_stage2_case_generation(request, settings, db, file_manager):
        sys.exit(1)

    print(f"\n{'='*60}")
    print("AUTO MODE: Both stages complete.")
    print(f"Request ID: {request_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

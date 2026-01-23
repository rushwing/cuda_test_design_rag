"""SQLite database for tracking test generation requests and reviews."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from cuda_test_rag.config import Settings

from .models import GenerationRequest, RequestStatus, ReviewRecord


class RequestDatabase:
    """SQLite database for request and review tracking."""

    def __init__(self, settings: Settings | None = None, db_path: str | None = None):
        self.settings = settings or Settings()
        self.db_path = Path(db_path or "./data/requests.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS generation_requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tester_uid TEXT NOT NULL,
                    tester_role TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    generation_config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    few_shot_source_paths_json TEXT,
                    generated_intents_path TEXT,
                    generated_tests_path TEXT,
                    llm_model_used TEXT,
                    generation_duration_seconds REAL,
                    num_cases_generated INTEGER,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS review_records (
                    review_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewer_uid TEXT NOT NULL,
                    reviewer_role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_notes TEXT,
                    accuracy_score INTEGER,
                    completeness_score INTEGER,
                    relevance_score INTEGER,
                    reviewed_tests_path TEXT,
                    approved_for_training INTEGER DEFAULT 0,
                    FOREIGN KEY (request_id) REFERENCES generation_requests(request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_requests_status ON generation_requests(status);
                CREATE INDEX IF NOT EXISTS idx_requests_tester ON generation_requests(tester_uid);
                CREATE INDEX IF NOT EXISTS idx_requests_created ON generation_requests(created_at);
                CREATE INDEX IF NOT EXISTS idx_reviews_request ON review_records(request_id);
                CREATE INDEX IF NOT EXISTS idx_reviews_training ON review_records(approved_for_training);
            """
            )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ==================== Request Operations ====================

    def create_request(self, request: GenerationRequest) -> str:
        """Create a new generation request."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO generation_requests (
                    request_id, created_at, updated_at, tester_uid, tester_role,
                    filters_json, generation_config_json, status,
                    few_shot_source_paths_json, generated_intents_path, generated_tests_path,
                    llm_model_used, generation_duration_seconds, num_cases_generated, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    request.request_id,
                    request.created_at.isoformat(),
                    request.updated_at.isoformat(),
                    request.tester_uid,
                    request.tester_role,
                    request.filters.model_dump_json(),
                    request.generation_config.model_dump_json(),
                    request.status.value,
                    json.dumps(request.few_shot_source_paths),
                    request.generated_intents_path,
                    request.generated_tests_path,
                    request.llm_model_used,
                    request.generation_duration_seconds,
                    request.num_cases_generated,
                    request.error_message,
                ),
            )
        return request.request_id

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM generation_requests WHERE request_id = ?", (request_id,)
            ).fetchone()

        if not row:
            return None

        return self._row_to_request(row)

    def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
        generated_tests_path: str | None = None,
        generated_intents_path: str | None = None,
        num_cases_generated: int | None = None,
        generation_duration_seconds: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update request status and related fields."""
        updates = ["status = ?", "updated_at = ?"]
        values = [status.value, datetime.utcnow().isoformat()]

        if generated_tests_path:
            updates.append("generated_tests_path = ?")
            values.append(generated_tests_path)
        if generated_intents_path:
            updates.append("generated_intents_path = ?")
            values.append(generated_intents_path)
        if num_cases_generated is not None:
            updates.append("num_cases_generated = ?")
            values.append(num_cases_generated)
        if generation_duration_seconds is not None:
            updates.append("generation_duration_seconds = ?")
            values.append(generation_duration_seconds)
        if error_message:
            updates.append("error_message = ?")
            values.append(error_message)

        values.append(request_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE generation_requests SET {', '.join(updates)} WHERE request_id = ?",
                values,
            )

    def list_requests(
        self,
        status: RequestStatus | None = None,
        tester_uid: str | None = None,
        limit: int = 100,
    ) -> list[GenerationRequest]:
        """List requests with optional filters."""
        query = "SELECT * FROM generation_requests WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status.value)
        if tester_uid:
            query += " AND tester_uid = ?"
            params.append(tester_uid)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_request(row) for row in rows]

    def _row_to_request(self, row: sqlite3.Row) -> GenerationRequest:
        """Convert a database row to a GenerationRequest."""
        from .models import GenerationConfig, TestFilters

        return GenerationRequest(
            request_id=row["request_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            tester_uid=row["tester_uid"],
            tester_role=row["tester_role"],
            filters=TestFilters.model_validate_json(row["filters_json"]),
            generation_config=GenerationConfig.model_validate_json(
                row["generation_config_json"]
            ),
            status=RequestStatus(row["status"]),
            few_shot_source_paths=json.loads(row["few_shot_source_paths_json"] or "[]"),
            generated_intents_path=row["generated_intents_path"],
            generated_tests_path=row["generated_tests_path"],
            llm_model_used=row["llm_model_used"],
            generation_duration_seconds=row["generation_duration_seconds"],
            num_cases_generated=row["num_cases_generated"],
            error_message=row["error_message"],
        )

    # ==================== Review Operations ====================

    def create_review(self, review: ReviewRecord) -> str:
        """Create a new review record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_records (
                    review_id, request_id, created_at, reviewer_uid, reviewer_role,
                    status, review_notes, accuracy_score, completeness_score,
                    relevance_score, reviewed_tests_path, approved_for_training
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    review.review_id,
                    review.request_id,
                    review.created_at.isoformat(),
                    review.reviewer_uid,
                    review.reviewer_role,
                    review.status.value,
                    review.review_notes,
                    review.accuracy_score,
                    review.completeness_score,
                    review.relevance_score,
                    review.reviewed_tests_path,
                    1 if review.approved_for_training else 0,
                ),
            )

            # Update the request status based on review
            conn.execute(
                "UPDATE generation_requests SET status = ?, updated_at = ? WHERE request_id = ?",
                (review.status.value, datetime.utcnow().isoformat(), review.request_id),
            )

        return review.review_id

    def get_reviews_for_request(self, request_id: str) -> list[ReviewRecord]:
        """Get all reviews for a request."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM review_records WHERE request_id = ? ORDER BY created_at DESC",
                (request_id,),
            ).fetchall()

        return [self._row_to_review(row) for row in rows]

    def get_training_ready_reviews(self) -> list[ReviewRecord]:
        """Get all reviews approved for training."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM review_records WHERE approved_for_training = 1"
            ).fetchall()

        return [self._row_to_review(row) for row in rows]

    def _row_to_review(self, row: sqlite3.Row) -> ReviewRecord:
        """Convert a database row to a ReviewRecord."""
        return ReviewRecord(
            review_id=row["review_id"],
            request_id=row["request_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            reviewer_uid=row["reviewer_uid"],
            reviewer_role=row["reviewer_role"],
            status=RequestStatus(row["status"]),
            review_notes=row["review_notes"] or "",
            accuracy_score=row["accuracy_score"],
            completeness_score=row["completeness_score"],
            relevance_score=row["relevance_score"],
            reviewed_tests_path=row["reviewed_tests_path"],
            approved_for_training=bool(row["approved_for_training"]),
        )

    # ==================== Statistics ====================

    def get_statistics(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            total_requests = conn.execute(
                "SELECT COUNT(*) FROM generation_requests"
            ).fetchone()[0]

            by_status = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM generation_requests GROUP BY status"
                ).fetchall()
            )

            total_reviews = conn.execute(
                "SELECT COUNT(*) FROM review_records"
            ).fetchone()[0]

            training_ready = conn.execute(
                "SELECT COUNT(*) FROM review_records WHERE approved_for_training = 1"
            ).fetchone()[0]

        return {
            "total_requests": total_requests,
            "requests_by_status": by_status,
            "total_reviews": total_reviews,
            "training_ready_count": training_ready,
        }

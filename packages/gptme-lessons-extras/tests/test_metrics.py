"""Tests for lesson metrics aggregation."""

import json
from pathlib import Path

import pytest
from gptme_lessons_extras.metrics import (
    LessonMetrics,
    MetricsAggregator,
    VersionMetrics,
)


class TestVersionMetrics:
    def test_to_dict_includes_all_fields(self) -> None:
        vm = VersionMetrics(
            version=2,
            uses=4,
            success_rate=0.85,
            contributors=["alice", "bob"],
            created="2025-01-15T12:00:00",
        )
        d = vm.to_dict()
        assert d["version"] == 2
        assert d["uses"] == 4
        assert d["success_rate"] == pytest.approx(0.85)
        assert d["contributors"] == ["alice", "bob"]
        assert d["created"] == "2025-01-15T12:00:00"


class TestLessonMetrics:
    def test_to_dict_includes_nested_versions(self) -> None:
        vm = VersionMetrics(
            version=1,
            uses=2,
            success_rate=0.9,
            contributors=["bob"],
            created="2025-01-10",
        )
        lm = LessonMetrics(
            lesson_id="my-lesson",
            total_uses=2,
            success_rate=0.9,
            adoption_count=1,
            versions=[vm],
            created="2025-01-10",
            last_updated="2025-01-10",
        )
        d = lm.to_dict()
        assert d["lesson_id"] == "my-lesson"
        assert d["total_uses"] == 2
        assert d["adoption_count"] == 1
        assert len(d["versions"]) == 1
        assert d["versions"][0]["version"] == 1


class TestMetricsAggregatorNoHistory:
    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        agg = MetricsAggregator(history_dir=tmp_path)
        assert agg.aggregate_lesson_metrics("nonexistent-lesson") is None

    def test_returns_empty_network_when_no_files(self, tmp_path: Path) -> None:
        agg = MetricsAggregator(history_dir=tmp_path)
        result = agg.aggregate_network_metrics()
        assert result == {}


class TestMetricsAggregatorWithHistory:
    def _write_history(
        self, history_dir: Path, lesson_id: str, n_versions: int, origin: str = "bob"
    ) -> None:
        history = {
            "created": "2025-01-01T00:00:00",
            "origin_agent": origin,
            "versions": [
                {
                    "version": i,
                    "timestamp": f"2025-01-0{i}T12:00:00",
                    "contributor": origin,
                }
                for i in range(1, n_versions + 1)
            ],
        }
        (history_dir / f"{lesson_id}.json").write_text(json.dumps(history))

    def test_aggregate_lesson_metrics_basic(self, tmp_path: Path) -> None:
        self._write_history(tmp_path, "my-lesson", n_versions=3)
        agg = MetricsAggregator(history_dir=tmp_path)
        metrics = agg.aggregate_lesson_metrics("my-lesson")
        assert metrics is not None
        assert metrics.lesson_id == "my-lesson"
        assert metrics.total_uses == 1 + 2 + 3  # version numbers summed
        assert len(metrics.versions) == 3
        assert metrics.adoption_count == 1  # single contributor
        assert 0.0 <= metrics.success_rate <= 1.0

    def test_aggregate_lesson_metrics_multiple_contributors(
        self, tmp_path: Path
    ) -> None:
        history = {
            "created": "2025-01-01T00:00:00",
            "origin_agent": "bob",
            "versions": [
                {
                    "version": 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "contributor": "alice",
                },
                {
                    "version": 2,
                    "timestamp": "2025-01-02T00:00:00",
                    "contributor": "bob",
                },
            ],
        }
        (tmp_path / "lesson-two.json").write_text(json.dumps(history))
        agg = MetricsAggregator(history_dir=tmp_path)
        metrics = agg.aggregate_lesson_metrics("lesson-two")
        assert metrics is not None
        assert metrics.adoption_count == 2

    def test_aggregate_lesson_metrics_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json{")
        agg = MetricsAggregator(history_dir=tmp_path)
        assert agg.aggregate_lesson_metrics("bad") is None

    def test_aggregate_network_scans_all_files(self, tmp_path: Path) -> None:
        self._write_history(tmp_path, "lesson-a", n_versions=2, origin="bob")
        self._write_history(tmp_path, "lesson-b", n_versions=1, origin="alice")
        agg = MetricsAggregator(history_dir=tmp_path)
        network = agg.aggregate_network_metrics()
        assert set(network.keys()) == {"lesson-a", "lesson-b"}

    def test_identify_best_practices_filters_low_adoption(self, tmp_path: Path) -> None:
        # lesson-a: only 1 adopter → should not appear with min_adoption=2
        self._write_history(tmp_path, "lesson-a", n_versions=5, origin="bob")
        agg = MetricsAggregator(history_dir=tmp_path)
        practices = agg.identify_best_practices(min_adoption=2)
        lesson_ids = [lid for lid, _ in practices]
        assert "lesson-a" not in lesson_ids

    def test_identify_best_practices_includes_wide_adoption(
        self, tmp_path: Path
    ) -> None:
        history = {
            "created": "2025-01-01",
            "origin_agent": "bob",
            "versions": [
                {
                    "version": 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "contributor": "alice",
                },
                {
                    "version": 2,
                    "timestamp": "2025-01-02T00:00:00",
                    "contributor": "bob",
                },
                {
                    "version": 3,
                    "timestamp": "2025-01-03T00:00:00",
                    "contributor": "charlie",
                },
            ],
        }
        (tmp_path / "wide-lesson.json").write_text(json.dumps(history))
        agg = MetricsAggregator(history_dir=tmp_path)
        practices = agg.identify_best_practices(min_adoption=2)
        lesson_ids = [lid for lid, _ in practices]
        assert "wide-lesson" in lesson_ids

    def test_generate_report_returns_string(self, tmp_path: Path) -> None:
        self._write_history(tmp_path, "lesson-x", n_versions=2, origin="bob")
        agg = MetricsAggregator(history_dir=tmp_path)
        report = agg.generate_report()
        assert isinstance(report, str)
        assert "# Lesson Network Metrics Report" in report

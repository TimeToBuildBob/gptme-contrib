"""Tests for generate.py — GEPA-lite evolution lesson generation."""

import json
from pathlib import Path
from unittest.mock import patch

from gptme_lessons_extras.generate import generate_lessons_with_evolution


def _make_analysis(tmp: Path, experiences: list[dict]) -> Path:
    """Write a minimal analysis JSON file and return its path."""
    # Ensure all experiences have a confidence field above the default threshold
    experiences = [{"confidence": 0.8, **exp} for exp in experiences]
    data = {
        "conversation_id": "test-conv-001",
        "experiences": experiences,
    }
    p = tmp / "analysis.json"
    p.write_text(json.dumps(data))
    return p


def _make_existing_lesson(tmp: Path, slug: str, title: str, context: str = "") -> Path:
    """Write a minimal existing lesson file."""
    lesson_dir = tmp / "lessons"
    lesson_dir.mkdir(parents=True, exist_ok=True)
    p = lesson_dir / f"{slug}.md"
    p.write_text(
        f"""---
match:
  keywords:
    - "test keyword"
status: active
---

# {title}

## Rule
Test rule.

## Context
{context}

## Detection
- Signal 1

## Pattern
```txt
example
```

## Outcome
- Benefit 1
"""
    )
    return p


def test_generate_checking_disabled(tmp_path):
    """Check existing disabled: no pre-generation similarity check."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Always log session decisions",
                "context": "Decisions lose context without logging",
                "confidence": 0.9,
            }
        ],
    )

    with (
        patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve,
        patch("gptme_lessons_extras.generate.save_lesson_draft") as mock_save,
    ):
        mock_evolve.return_value = (
            "# Test Lesson\n\n## Rule\nAlways log.",
            {"scores": {"clarity": 0.8, "relevance": 0.9}},
            [],
        )
        mock_save.return_value = Path("/fake/output/always-log-session.md")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            check_existing=False,
            verbose=False,
        )

    assert len(result) == 1, "Should have generated the lesson"
    mock_evolve.assert_called_once()


def test_generate_no_existing_matches(tmp_path):
    """No similar existing lessons: lesson should be generated."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Always log session decisions",
                "context": "Decisions lose context without logging",
                "confidence": 0.9,
            }
        ],
    )
    existing_dir = tmp_path / "existing_lessons"
    existing_dir.mkdir()
    # Create an unrelated lesson
    _make_existing_lesson(
        existing_dir, "unrelated", "Something completely different", "Other context"
    )

    with (
        patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve,
        patch("gptme_lessons_extras.generate.save_lesson_draft") as mock_save,
    ):
        mock_evolve.return_value = (
            "# Test Lesson\n\n## Rule\nAlways log.",
            {"scores": {"clarity": 0.8, "relevance": 0.9}},
            [],
        )
        mock_save.return_value = Path("/fake/output/always-log-session.md")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            check_existing=True,
            existing_lessons_dir=existing_dir,
            verbose=False,
        )

    assert len(result) == 1, "Should generate when no duplicate found"
    mock_evolve.assert_called_once()


def test_generate_duplicate_skip_default(tmp_path):
    """Duplicate found with skip_duplicates=True: lesson is skipped."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Always log session decisions",
                "context": "Decisions lose context without logging",
            }
        ],
    )
    existing_dir = tmp_path / "existing_lessons"
    existing_dir.mkdir()
    # Create a very similar lesson — same title
    _make_existing_lesson(
        existing_dir,
        "always-log",
        "Always log session decisions",
        "Decisions lose context without logging",
    )

    with patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve:
        mock_evolve.return_value = (
            "# Test Lesson\n\n## Rule\nAlways log.",
            {"scores": {"clarity": 0.8, "relevance": 0.9}},
            [],
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            check_existing=True,
            existing_lessons_dir=existing_dir,
            similarity_threshold=0.6,
            skip_duplicates=True,
            verbose=False,
        )

    assert (
        len(result) == 0
    ), "Should skip when duplicate found with skip_duplicates=True"
    mock_evolve.assert_not_called()


def test_generate_duplicate_warn_only(tmp_path):
    """Duplicate found with skip_duplicates=False: lesson is still generated."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Always log session decisions",
                "context": "Decisions lose context without logging",
            }
        ],
    )
    existing_dir = tmp_path / "existing_lessons"
    existing_dir.mkdir()
    _make_existing_lesson(
        existing_dir,
        "always-log",
        "Always log session decisions",
        "Decisions lose context without logging",
    )

    with (
        patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve,
        patch("gptme_lessons_extras.generate.save_lesson_draft") as mock_save,
    ):
        mock_evolve.return_value = (
            "# Test Lesson\n\n## Rule\nAlways log.",
            {"scores": {"clarity": 0.8, "relevance": 0.9}},
            [],
        )
        mock_save.return_value = Path("/fake/output/always-log-session.md")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            check_existing=True,
            existing_lessons_dir=existing_dir,
            similarity_threshold=0.6,
            skip_duplicates=False,
            verbose=False,
        )

    assert (
        len(result) == 1
    ), "Should generate even when duplicate found with skip_duplicates=False"
    mock_evolve.assert_called_once()


def test_generate_low_confidence_skipped(tmp_path):
    """Experiences below min_confidence are filtered out."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Low confidence lesson",
                "context": "Not very sure about this",
                "confidence": 0.3,
            }
        ],
    )

    with patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            min_confidence=0.6,
            verbose=False,
        )

    assert len(result) == 0, "Should not generate for low-confidence experiences"
    mock_evolve.assert_not_called()


def test_generate_learnable_moments_fallback(tmp_path):
    """Falls back to metadata.learnable_moments when no top-level experiences."""
    data = {
        "conversation_id": "test-conv-002",
        "experiences": [],
        "metadata": {
            "learnable_moments": [
                {
                    "title": "Fallback lesson",
                    "context": "From old format",
                    "confidence": 0.9,
                }
            ]
        },
    }
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(data))

    with (
        patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve,
        patch("gptme_lessons_extras.generate.save_lesson_draft") as mock_save,
    ):
        mock_evolve.return_value = (
            "# Test Lesson\n\n## Rule\nFallback.",
            {"scores": {"clarity": 0.8}},
            [],
        )
        mock_save.return_value = Path("/fake/output/fallback.md")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            verbose=False,
        )

    assert len(result) == 1, "Should generate from learnable_moments fallback"
    mock_evolve.assert_called_once()


def test_generate_judge_threshold_below(tmp_path):
    """Lesson below judge_threshold: not included."""
    analysis = _make_analysis(
        tmp_path,
        [
            {
                "title": "Mediocre lesson",
                "context": "Some context",
                "confidence": 0.9,
            }
        ],
    )

    with (
        patch("gptme_lessons_extras.generate.gepa_lite_evolve") as mock_evolve,
        patch("gptme_lessons_extras.generate.save_lesson_draft") as mock_save,
    ):
        mock_evolve.return_value = (
            "# Mediocre Lesson\n\n## Rule\nMeh.",
            {"scores": {"clarity": 0.3, "relevance": 0.4}},
            [],
        )
        mock_save.return_value = Path("/fake/output/mediocre.md")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generate_lessons_with_evolution(
            analysis_file=analysis,
            output_dir=output_dir,
            judge_threshold=0.6,
            verbose=False,
        )

    assert len(result) == 0, "Should skip lessons below judge_threshold"

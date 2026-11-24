# Repro tests — gptme-contrib rag/codegraph verified bug hunt (2026-11-24, session 6017)

Every test failed on upstream master 853640df. Findings table: tasks/verified-bug-hunt-2026-11-24-contrib-rag-codegraph.md. To re-run: write a block to a .py file and run it with the contrib venv from packages/gptme-rag (rag) or the repo root (codegraph). r6 needs its conftest.py block next to it.

## me/test_deleted_file_stale.py

```python
"""index never removes chunks of files deleted from the indexed directory."""
from click.testing import CliRunner

from gptme_rag.cli import cli
from gptme_rag.indexing.indexer import Indexer


def _index(runner, index_dir, src):
    return runner.invoke(
        cli, ["index", "--persist-dir", str(index_dir), "--embedding-function", "default", str(src)]
    )


def test_reindex_drops_chunks_of_deleted_file(tmp_path):
    runner = CliRunner()
    index_dir = tmp_path / "index"
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("kept file about apples")
    gone = src / "gone.txt"
    gone.write_text("deleted file about zebras")
    assert _index(runner, index_dir, src).exit_code == 0
    gone.unlink()
    second = _index(runner, index_dir, src)
    assert second.exit_code == 0, second.output
    idx = Indexer(persist_directory=index_dir, enable_persist=True, embedding_function="default")
    sources = {d.metadata.get("source") for d in idx.get_all_documents()}
    assert str(gone) not in sources, sources
```

## r1/test_gitignore_bugs.py

```python
"""Non-git fallback: .gitignore directory patterns ("build/") must be honoured."""
import uuid

from gptme_rag.indexing.indexer import Indexer


def _idx(tmp_path):
    return Indexer(
        persist_directory=tmp_path / "index",
        enable_persist=True,
        embedding_function="default",
        collection_name=f"c{uuid.uuid4().hex[:8]}",
    )


def test_gitignore_trailing_slash_dir_pattern_ignored(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "build").mkdir(parents=True)
    (corpus / "keep.md").write_text("keep")
    (corpus / "build" / "out.md").write_text("generated")
    (corpus / ".gitignore").write_text("build/\n")
    files = {p.relative_to(corpus.resolve()).as_posix() for p in _idx(tmp_path)._get_valid_files(corpus)}
    assert "build/out.md" not in files, files


def test_gitignore_anchored_pattern_ignored(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "keep.md").write_text("keep")
    (corpus / "secret.md").write_text("secret")
    (corpus / ".gitignore").write_text("/secret.md\n")
    files = {p.relative_to(corpus.resolve()).as_posix() for p in _idx(tmp_path)._get_valid_files(corpus)}
    assert "secret.md" not in files, files
```

## r1/test_indexer_bugs.py

```python
"""Bug-hunt reproductions for gptme_rag.indexing.indexer / document_processor."""

from __future__ import annotations

import uuid

import pytest

from gptme_rag.indexing.document import Document
from gptme_rag.indexing.document_processor import DocumentProcessor
from gptme_rag.indexing.indexer import Indexer


def _words(n: int) -> str:
    # Distinct words, one token each-ish, newline-free so overlap is visible.
    return " ".join(f"w{i}" for i in range(n))


@pytest.fixture
def idx(tmp_path):
    return Indexer(
        persist_directory=tmp_path / "index",
        enable_persist=True,
        embedding_function="default",
        chunk_size=50,
        chunk_overlap=10,
        collection_name=f"c{uuid.uuid4().hex[:8]}",
    )


def test_index_directory_emptied_file_leaves_no_entry(idx, tmp_path):
    """An emptied file must disappear from the index, not become an empty doc."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "note.md"
    f.write_text("hello world, some real content")
    idx.index_directory(corpus, glob_pattern="**/*.md")
    f.write_text("")
    idx.index_directory(corpus, glob_pattern="**/*.md")

    got = idx.collection.get(where={"source": str(f.resolve())})
    assert got["ids"] == [], f"tombstone stored as a document: {got['documents']!r}"


def test_delete_document_by_base_id_deletes_chunks(idx, tmp_path):
    """delete_document(base_id) is documented to delete the doc and all its chunks."""
    f = tmp_path / "big.txt"
    f.write_text(_words(200))
    idx.index_file(f)
    ids = idx.collection.get()["ids"]
    assert len(ids) > 1
    base_id = ids[0].split("#chunk")[0]

    assert idx.delete_document(base_id) is True
    remaining = idx.collection.get()["ids"]
    assert remaining == [], f"{len(remaining)} chunks survived delete_document({base_id!r})"


def test_reconstruct_document_matches_original(idx, tmp_path):
    """reconstruct_document should rebuild the original file content."""
    text = _words(200)
    f = tmp_path / "big.txt"
    f.write_text(text)
    idx.index_file(f)
    base_id = idx.collection.get()["ids"][0].split("#chunk")[0]

    doc = idx.reconstruct_document(base_id)
    assert doc.content == text


def test_get_document_chunks_ordered_by_chunk_index(idx, tmp_path):
    """get_document_chunks docstring: 'ordered by chunk index'."""
    f = tmp_path / "big.txt"
    f.write_text(_words(200))
    docs = list(Document.from_file(f.resolve(), processor=idx.processor))
    assert len(docs) > 2
    # Add in reverse order (e.g. batches retried / added out of order).
    idx.add_documents(list(reversed(docs)))
    base_id = docs[0].doc_id.split("#chunk")[0]

    chunks = idx.get_document_chunks(base_id)
    order = [c.metadata["chunk_index"] for c in chunks]
    assert order == sorted(order)


def test_get_document_chunks_ordered_many_chunks(idx, tmp_path):
    """Normal insertion order with >10 chunks: must still be ordered by index."""
    f = tmp_path / "big.txt"
    f.write_text(_words(600))
    idx.index_file(f)
    base_id = idx.collection.get()["ids"][0].split("#chunk")[0]
    chunks = idx.get_document_chunks(base_id)
    order = [c.metadata["chunk_index"] for c in chunks]
    assert len(order) > 10
    assert order == sorted(order)


def test_total_chunks_matches_emitted_chunks():
    """total_chunks metadata must equal the number of chunks actually yielded."""
    p = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    mismatches = []
    for n in range(51, 400):
        tokens = list(range(n))
        text = p.encoding.decode(p.encoding.encode(_words(400))[:n])
        chunks = list(p.process_text(text))
        if len(p.encoding.encode(text)) != n:
            continue
        totals = {c["metadata"]["total_chunks"] for c in chunks}
        if totals != {len(chunks)}:
            mismatches.append((n, len(chunks), totals))
        del tokens
    assert not mismatches, f"first mismatches (n_tokens, yielded, total_chunks): {mismatches[:3]}"


def test_chunks_do_not_contain_replacement_chars():
    """Chunking must not split multi-byte characters into U+FFFD garbage."""
    p = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    text = "日本語のテキストです。" * 40 + " 🎉🚀" * 30
    assert "�" not in text
    bad = [c["metadata"]["chunk_index"] for c in p.process_text(text) if "�" in c["text"]]
    assert not bad, f"chunks with U+FFFD: {bad}"
```

## r2/test_cache_bugs.py

```python
"""Reproductions for SmartRAGCache size-accounting / eviction defects."""

from datetime import datetime, timedelta

from gptme_rag.cache import CacheEntry, CacheKey, SmartRAGCache


def _entry(content: str = "x", age_s: float = 0.0) -> CacheEntry:
    now = datetime.now() - timedelta(seconds=age_s)
    return CacheEntry(
        document_contents=[content],
        document_metadatas=[{"source": "a.md"}],
        document_ids=["id1"],
        distances=[0.1],
        created_at=now,
        last_accessed=now,
        access_count=0,
        workspace_mtime=0.0,
        index_mtime=0.0,
        embedding_time_ms=0.0,
        result_count=1,
    )


def test_ttl_expiry_releases_size_accounting():
    cache = SmartRAGCache(ttl_seconds=60)
    key = CacheKey.from_search("q")
    cache.put(key, _entry(age_s=120))  # already stale
    assert cache.get(key) is None  # TTL eviction
    assert cache.get_stats()["entries"] == 0
    assert cache.get_stats()["total_size_bytes"] == 0


def test_ttl_expiry_does_not_shrink_effective_capacity():
    size = _entry().size_bytes()
    cache = SmartRAGCache(ttl_seconds=60, max_memory_bytes=3 * size)
    for i in range(3):
        k = CacheKey.from_search(f"old{i}")
        cache.put(k, _entry(age_s=120))
        assert cache.get(k) is None  # expired + removed
    # Cache is empty; two fresh entries fit easily in a 3-entry budget.
    k1, k2 = CacheKey.from_search("new1"), CacheKey.from_search("new2")
    cache.put(k1, _entry())
    cache.put(k2, _entry())
    assert cache.get(k1) is not None, "fresh entry evicted by leaked size accounting"


def test_memory_limit_counts_cached_document_contents():
    """SmartRAGCache advertises memory-aware eviction (max_memory_bytes); the
    cached full document contents must count toward that budget."""
    cache = SmartRAGCache(max_memory_bytes=1_000_000)
    big = "y" * 500_000
    for i in range(10):
        cache.put(CacheKey.from_search(f"q{i}"), _entry(content=big))
    held = sum(len(c) for e in cache.cache.values() for c in e.document_contents)
    assert held <= 1_000_000, f"cache holds {held} bytes of content, limit 1_000_000"
```

## r2/test_watcher_bugs.py

```python
"""Reproductions for gptme-rag watcher stale-state defects.

Uses a real Indexer (chroma "default" ONNX embedder, works offline) where the
assertion is about what the index/search actually returns, and a small fake
indexer where only call ordering matters.
"""

import os
import shutil
import time
from pathlib import Path

import pytest
from watchdog.events import DirDeletedEvent, FileDeletedEvent, FileMovedEvent

from gptme_rag.indexing.indexer import Indexer
from gptme_rag.indexing.watcher import FileWatcher, IndexEventHandler


@pytest.fixture
def real_indexer(tmp_path, request):
    name = "c" + str(abs(hash(request.node.name)))
    idx = Indexer(
        persist_directory=tmp_path / "index",
        enable_persist=True,
        embedding_function="default",
        collection_name=name,
    )
    idx.reset_collection()
    return idx


def _contents_for(indexer: Indexer, source: Path) -> list[str]:
    got = indexer.collection.get(where={"source": str(source)})
    return list(got["documents"] or [])


def _all_sources(indexer: Indexer) -> set[str]:
    got = indexer.collection.get()
    return {m["source"] for m in got["metadatas"] or []}


def test_move_over_existing_file_drops_old_content(tmp_path, real_indexer):
    """Atomic save (write temp, rename over target) must replace the target's
    indexed content, not add the new content alongside the old chunks."""
    watched = tmp_path / "w"
    watched.mkdir()
    target = (watched / "notes.md").resolve()
    target.write_text("old content about bananas")
    real_indexer.index_file(target)
    assert any("bananas" in c for c in _contents_for(real_indexer, target))

    tmp = watched / "notes.md.tmp"
    tmp.write_text("new content about cherries")
    os.replace(tmp, target)

    handler = IndexEventHandler(real_indexer)
    handler.on_moved(FileMovedEvent(str(tmp), str(target)))

    contents = _contents_for(real_indexer, target)
    assert any("cherries" in c for c in contents)
    assert not any("bananas" in c for c in contents), contents


def test_rename_out_of_pattern_does_not_leave_stale_search_cache(tmp_path, real_indexer):
    """After a watched file is renamed to a name outside the pattern, search
    must not keep returning it from the result cache."""
    watched = tmp_path / "w"
    watched.mkdir()
    src = (watched / "doc.md").resolve()
    src.write_text("zebra stripes savanna")
    real_indexer.index_file(src)

    docs, _, _ = real_indexer.search("zebra stripes savanna", n_results=1)
    assert docs and docs[0].metadata["source"] == str(src)

    dest = (watched / "doc.txt").resolve()
    src.rename(dest)
    handler = IndexEventHandler(real_indexer, pattern="**/*.md")
    handler.on_moved(FileMovedEvent(str(src), str(dest)))

    assert str(src) not in _all_sources(real_indexer)  # index itself is updated
    docs, _, _ = real_indexer.search("zebra stripes savanna", n_results=1)
    assert [d.metadata["source"] for d in docs] == [], "search served stale cached result"


def test_directory_moved_out_of_watch_is_removed_from_index(tmp_path, real_indexer):
    """`mv watched/sub elsewhere/` produces only a DirDeletedEvent (verified
    with the inotify observer); the files under it must leave the index."""
    watched = tmp_path / "w"
    (watched / "sub").mkdir(parents=True)
    f = (watched / "sub" / "a.md").resolve()
    f.write_text("directory move payload")
    real_indexer.index_file(f)
    assert str(f) in _all_sources(real_indexer)

    shutil.move(str(watched / "sub"), str(tmp_path / "elsewhere"))
    handler = IndexEventHandler(real_indexer)
    handler._update_delay = 0.05
    handler.on_deleted(DirDeletedEvent(str(watched / "sub")))
    time.sleep(0.5)

    assert str(f) not in _all_sources(real_indexer)


def test_watch_start_keeps_documents_from_other_directories(tmp_path, real_indexer):
    """`gptme-rag watch` documents that the existing collection is kept
    unchanged (cli.py watch: 'silently deleting the entire collection ... would
    cause permanent data loss'). Starting the watcher must not wipe documents
    indexed from other directories into the same persist dir."""
    other = tmp_path / "other"
    other.mkdir()
    keep = (other / "keep.md").resolve()
    keep.write_text("previously indexed knowledge")
    real_indexer.index_file(keep)

    watched = tmp_path / "w"
    watched.mkdir()
    (watched / "x.md").write_text("watched file")

    watcher = FileWatcher(real_indexer, [str(watched)], update_delay=0)
    watcher.startup_delay = 0
    watcher.start()
    try:
        assert str(keep) in _all_sources(real_indexer)
    finally:
        watcher.stop()


class _FakeIndex:
    """Fake indexer that tracks which sources are present."""

    class _Cache:
        def clear(self):
            pass

    def __init__(self, persist):
        self.persist_directory = persist
        self.cache = self._Cache()
        self.sources: set[str] = set()

    def delete_documents(self, where):
        self.sources.discard(where["source"])

    def index_file(self, path):
        self.sources.add(str(path))
        return 1

    def search(self, query, n_results=5):
        from types import SimpleNamespace

        docs = [SimpleNamespace(metadata={"source": s}) for s in sorted(self.sources)]
        return docs, [], None


def test_move_onto_path_with_pending_deletion(tmp_path):
    """`rm b.md; mv a.md b.md` within the debounce window: the pending
    deletion timer for b.md must not remove the freshly moved-in file."""
    a = (tmp_path / "a.md").resolve()
    b = (tmp_path / "b.md").resolve()
    a.write_text("A")
    b.write_text("B")
    idx = _FakeIndex(tmp_path / "index")
    idx.sources = {str(a), str(b)}
    handler = IndexEventHandler(idx)
    handler._update_delay = 0.1

    b.unlink()
    handler.on_deleted(FileDeletedEvent(str(b)))
    os.replace(a, b)
    handler.on_moved(FileMovedEvent(str(a), str(b)))
    assert str(b) in idx.sources  # indexed synchronously by on_moved
    time.sleep(0.5)  # let the stale deletion timer fire

    assert b.exists()
    assert str(b) in idx.sources, "pending deletion removed the moved-in file"
```

## r3/test_path_filter_glob.py

```python
from pathlib import Path

from gptme_rag.indexing.document import Document
from gptme_rag.indexing.indexer import Indexer


def _matches(source: str, pattern: str) -> bool:
    doc = Document(content="", metadata={"source": source}, doc_id="x")
    # _matches_paths does not touch instance state
    return Indexer._matches_paths(None, doc, paths=None, path_filters=(pattern,))  # type: ignore[arg-type]


def test_single_star_does_not_cross_directories():
    # Docstring distinguishes "src/*.py" from "src/**/*.py" ("Python files in src
    # directory" needs **); a single * must not match nested subdirectories.
    assert _matches("/proj/src/mod.py", "src/*.py")
    assert not _matches("/proj/src/pkg/deep/mod.py", "src/*.py")
```

## r3/test_search_paths.py

```python
import json

import pytest
from click.testing import CliRunner

from gptme_rag.cli import cli
from gptme_rag.indexing.document import Document
from gptme_rag.indexing.indexer import Indexer


def _build(tmp_path):
    a = tmp_path / "data" / "a"
    b = tmp_path / "data" / "b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    idx = Indexer(
        persist_directory=tmp_path / "index",
        enable_persist=True,
        embedding_function="default",  # chroma's cached ONNX MiniLM, no download
        chunk_size=500,
        chunk_overlap=0,
    )
    docs = [
        Document(
            content=f"Python programming language tutorial number {i}",
            metadata={"source": str(b / f"py{i}.txt")},
            doc_id=f"b{i}",
        )
        for i in range(8)
    ]
    docs.append(
        Document(
            content="Recipe for a vegetable soup with carrots",
            metadata={"source": str(a / "soup.txt")},
            doc_id="a0",
        )
    )
    idx.add_documents(docs)
    return idx, a, b


def test_paths_restriction_returns_docs_under_path(tmp_path):
    idx, a, _b = _build(tmp_path)
    docs, _d, _ = idx.search("python programming", paths=[a], n_results=1)
    # A document exists under `a`; restricting to `a` should return it, not nothing.
    assert [d.metadata["source"] for d in docs] == [str(a / "soup.txt")]


def test_cli_filter_without_paths_searches_whole_index(tmp_path, monkeypatch):
    idx, _a, b = _build(tmp_path)
    del idx
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runner = CliRunner()
    base = ["search", "python programming", "--persist-dir", str(tmp_path / "index"), "--json"]
    plain = json.loads(runner.invoke(cli, base, catch_exceptions=False).output)
    assert plain["total_results"] > 0
    filtered = json.loads(
        runner.invoke(cli, base + ["--filter", "*.txt"], catch_exceptions=False).output
    )
    # Every indexed doc is *.txt, so adding --filter '*.txt' must not drop results.
    assert filtered["total_results"] == plain["total_results"]


def test_where_combined_with_path_filters_raises(tmp_path):
    idx, _a, _b = _build(tmp_path)
    # Any non-source where clause + path_filters -> chroma rejects the 2-key dict.
    docs, _d, _ = idx.search(
        "python programming", n_results=2, where={"chunk_index": 0}, path_filters=("*.txt",)
    )
    assert len(docs) == 2


def test_where_source_not_overwritten_by_path_filters(tmp_path):
    idx, _a, b = _build(tmp_path)
    only = str(b / "py0.txt")
    docs, _d, _ = idx.search(
        "python programming", n_results=5, where={"source": only}, path_filters=("*.txt",)
    )
    assert {d.metadata["source"] for d in docs} == {only}
```

## r3/test_task_frontmatter.py

```python
from gptme_rag.task_retrieval import load_task_documents


def _load(tmp_path, text):
    (tmp_path / "t.md").write_text(text)
    [doc] = load_task_documents(tmp_path)
    return doc.metadata


def test_empty_value_key_does_not_swallow_state(tmp_path):
    # An empty-valued key (common: `assigned_to:` / `waiting_for:`) right before
    # `state:` must not consume the next line as its value.
    md = _load(tmp_path, "---\nassigned_to:\nstate: done\n---\n# My task\n")
    assert md["task_state"] == "done"


def test_frontmatter_only_file_without_trailing_newline(tmp_path):
    md = _load(tmp_path, "---\nstate: done\ntitle: Ship it\n---")
    assert md["task_state"] == "done"
    assert md["title"] == "Ship it"
```

## r4/test_findings.py

```python
"""Verified defects in gptme_rag lesson_matcher / observability (bughunt r4)."""
import math
from pathlib import Path

import pytest

from gptme_rag import lesson_matcher as lm
from gptme_rag.lesson_matcher import (
    filter_by_repo,
    normalize_repo_slug,
    scan_lessons,
    score_lessons,
)
from gptme_rag.observability import log_injection, summarize_injections


@pytest.fixture(autouse=True)
def _nocache(monkeypatch):
    monkeypatch.setenv("GPTME_LESSON_SCAN_CACHE", "off")
    lm.clear_scan_cache()
    yield
    lm.clear_scan_cache()


def _scan_one(tmp_path: Path, text: str):
    d = tmp_path / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.md").write_text(text)
    return scan_lessons([d])


# F1: nested match.keywords scalar string is not comma-split under PyYAML,
# although the regex fallback (and top-level keywords, session_categories,
# repos) split it.  Adding an unrelated YAML error flips the result.
def test_nested_keywords_scalar_split_like_fallback(tmp_path):
    good = "---\nmatch:\n  keywords: git push, force push\n---\n# A\n"
    # Same lesson, but YAML parse fails (unquoted ': ' in description) -> regex fallback
    broken = "---\ndescription: Fix: thing\nmatch:\n  keywords: git push, force push\n---\n# A\n"
    fallback_kw = _scan_one(tmp_path / "fallback", broken)[0]["keywords"]
    yaml_kw = _scan_one(tmp_path / "yaml", good)[0]["keywords"]
    assert fallback_kw == ["git push", "force push"]
    assert yaml_kw == fallback_kw


def test_nested_keywords_scalar_matches_prompt(tmp_path):
    lessons = _scan_one(tmp_path, "---\nmatch:\n  keywords: git push, force push\n---\n# A\n")
    assert score_lessons(lessons, "please git push this", use_bm25=False)


# F2: non-string YAML items in match.keywords (e.g. HTTP status 429) are
# silently dropped by _dedupe_strings; a lesson whose only keyword is numeric
# is skipped entirely.  Top-level keywords and the regex fallback keep them.
def test_numeric_nested_keyword_kept(tmp_path):
    lessons = _scan_one(tmp_path, "---\nmatch:\n  keywords: [429]\n---\n# Rate limits\n")
    assert lessons, "lesson with numeric keyword was dropped entirely"
    assert score_lessons(lessons, "got HTTP 429 from the API", use_bm25=False)


# F3: a top-level scalar `patterns:` regex containing a comma is split on the
# comma by _string_list, corrupting quantifiers like {3,4}.
def test_top_level_pattern_comma_not_split(tmp_path):
    lessons = _scan_one(tmp_path, "---\npatterns: 'exit code \\d{3,4}'\n---\n# A\n")
    assert lessons[0]["patterns"] == [r"exit code \d{3,4}"]
    assert score_lessons(lessons, "failed with exit code 1234", use_bm25=False)


# F4: ssh:// remote URLs (an SSH remote string, as `git remote get-url` can
# return) normalise to None, so filter_by_repo fail-closes repo-scoped lessons.
def test_ssh_scheme_remote_normalised():
    assert normalize_repo_slug("ssh://git@github.com/gptme/gptme.git") == "gptme/gptme"
    lesson = {"repos": ["gptme/gptme"], "repos_restricted": True}
    assert filter_by_repo([lesson], "ssh://git@github.com/gptme/gptme.git") == [lesson]


# F5: NaN scores (e.g. cosine similarity of a zero vector) bypass _as_score,
# get written as bare NaN, and permanently poison avg_top_score.
def test_nan_score_does_not_poison_summary(tmp_path):
    log = tmp_path / "inj.jsonl"
    log_injection(log, "q", [{"similarity": float("nan")}], backend="t", session_id="s", harness="h")
    log_injection(log, "q", [{"similarity": 0.5}], backend="t", session_id="s", harness="h")
    stats = summarize_injections(log)
    assert not math.isnan(stats.avg_top_score)
```

## r4/test_secondary.py

```python
"""Reproduced but lower-confidence (intent arguable) items."""
import json
from datetime import datetime, timezone
from pathlib import Path

from gptme_rag.knowledge_source import collect_knowledge_entry_documents
from gptme_rag.lesson_matcher import extract_frontmatter
from gptme_rag.observability import assess_index_health, log_injection


def test_empty_on_disk_slice_is_not_missing():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    h = assess_index_health(built_at=now, indexed_count=100, on_disk_count=100,
                            slices={"x": (0, 0)}, now=now)
    assert h.status == "ok", h.reasons


def test_invalid_entry_type_skipped_like_gptme(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"id": "12345678-1234-5678-1234-567812345678", "problem": "p",
                             "resolution": "r", "tags": [], "created_at": "2026-01-01T00:00:00+00:00",
                             "entry_type": "bogus"}) + "\n")
    assert collect_knowledge_entry_documents(p) == []


def test_bom_frontmatter_parsed():
    fm, _ = extract_frontmatter("﻿---\nmatch:\n  keywords: [foo]\n---\n# A\n")
    assert fm.get("match", {}).get("keywords") == ["foo"]


def test_triple_dash_inside_frontmatter_value():
    fm, _ = extract_frontmatter("---\nmatch:\n  keywords: ['--- a/']\n---\n# A\n")
    assert fm.get("match", {}).get("keywords") == ["--- a/"]


def test_path_valued_hit_field_is_logged(tmp_path):
    rec = log_injection(tmp_path / "l.jsonl", "q", [{"path": Path("/a.md"), "similarity": 0.5}],
                        backend="t", session_id="s", harness="h")
    assert rec is not None
```

## r5/test_r5_codegraph.py

```python
"""Verified-bug tests for gptme_codegraph.core (bughunt r5)."""

import os
import subprocess
from pathlib import Path

import pytest
from gptme_codegraph import core
from gptme_codegraph.core import (
    SqliteIndexCache,
    build_cross_file_call_graph,
    build_index,
    extract_symbols,
)


@pytest.fixture(autouse=True)
def _isolate_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_CODEGRAPH_STATE_DIR", tmp_path / "state")


def _git(d: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(d), *args],
        check=True,
        capture_output=True,
    )


def _repo(d: Path) -> Path:
    d.mkdir()
    _git(d, "init", "-q")
    return d


# --- Finding 1: tracked-but-deleted file (unstaged rm) crashes build_index ---
def test_build_index_survives_tracked_file_deleted_from_worktree(tmp_path):
    d = _repo(tmp_path / "repo")
    (d / "a.py").write_text("def a():\n    pass\n")
    (d / "b.py").write_text("def b():\n    pass\n")
    _git(d, "add", ".")
    (d / "b.py").unlink()  # user deleted the file, not yet `git rm`-ed
    index = build_index(d)
    assert index.has("a")
    assert not index.has("b")


# --- Finding 1b: broken symlink crashes build_index (non-git dir) ---
def test_build_index_survives_dangling_symlink(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("def a():\n    pass\n")
    os.symlink(d / "gone.py", d / "link.py")
    index = build_index(d)
    assert index.has("a")


# --- Finding 2: non-ASCII file names come back C-quoted from git ls-files ---
def test_iter_source_files_handles_non_ascii_names(tmp_path):
    d = _repo(tmp_path / "repo")
    (d / "café.py").write_text("def cafe():\n    pass\n")
    _git(d, "add", ".")
    files = core._iter_source_files(d)
    assert files == [d / "café.py"]
    assert build_index(d).has("cafe")


# --- Finding 3: untracked (not ignored) files are invisible, cache never invalidates ---
def test_untracked_new_file_is_indexed(tmp_path):
    d = _repo(tmp_path / "repo")
    (d / "a.py").write_text("def a():\n    pass\n")
    _git(d, "add", ".")
    (d / "new.py").write_text("def brand_new():\n    pass\n")
    assert build_index(d).has("brand_new")


def test_sqlite_cache_goes_stale_when_untracked_file_added(tmp_path):
    d = _repo(tmp_path / "repo")
    (d / "a.py").write_text("def a():\n    pass\n")
    _git(d, "add", ".")
    cache = SqliteIndexCache(str(d))
    cache.save(build_index(d))
    assert cache.is_fresh()
    (d / "new.py").write_text("def brand_new():\n    pass\n")
    assert not cache.is_fresh(), "new source file must invalidate the cache"
    cache.close()


# --- Finding 4: cross-file graph collapses same-named symbols onto one qid ---
def test_cross_file_graph_keeps_same_named_functions_distinct(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("def helper_a():\n    pass\n\ndef main():\n    helper_a()\n")
    (d / "b.py").write_text("def helper_b():\n    pass\n\ndef main():\n    helper_b()\n")
    callees, callers = build_cross_file_call_graph(build_index(d), d)
    assert callees.get("a::main") == {"a::helper_a"}
    assert callees.get("b::main") == {"b::helper_b"}


def test_cross_file_graph_prefers_same_file_definition(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("def helper():\n    pass\n")
    (d / "b.py").write_text("def helper():\n    pass\n\ndef run():\n    helper()\n")
    callees, _ = build_cross_file_call_graph(build_index(d), d)
    assert callees.get("b::run") == {"b::helper"}


# --- Finding 5: Python defs under if/try blocks are not extracted ---
def test_python_conditional_definitions_extracted(tmp_path):
    f = tmp_path / "compat.py"
    f.write_text(
        "import sys\n"
        "if sys.version_info >= (3, 11):\n"
        "    def fast():\n        pass\n"
        "try:\n"
        "    from ujson import dumps\n"
        "except ImportError:\n"
        "    def dumps(obj):\n        pass\n"
        "class K:\n"
        "    if True:\n"
        "        def m(self):\n            pass\n"
    )
    names = {(s.name, s.kind) for s in extract_symbols(f)}
    assert ("fast", "function") in names
    assert ("dumps", "function") in names
    assert ("m", "method") in names
```

## r6/conftest.py

```python
import pytest


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    from gptme_codegraph import commit_map, core
    import gptme_codegraph.mcp_server as mcp_mod

    monkeypatch.setattr(commit_map, "_CACHE_DIR", tmp_path / "cg-cache")
    monkeypatch.setattr(core, "_CODEGRAPH_STATE_DIR", tmp_path / "cg-state")
    mcp_mod._index_cache.clear()
    yield
    mcp_mod._index_cache.clear()
```

## r6/test_commit_map_bugs.py

```python
"""Reproductions for gptme_codegraph.commit_map defects."""

import os
import subprocess
from pathlib import Path

from gptme_codegraph import commit_map

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=GIT_ENV,
    ).stdout.strip()


def _symbol_names(map_data) -> set[str]:
    names = set()
    for f in map_data["files"]:
        for s in f.get("outline", []):
            names.add(s["name"])
            for m in s.get("methods", []) or []:
                names.add(m["name"])
    return names


def _bump(path: Path, text: str) -> None:
    st = path.stat()
    path.write_text(text)
    # make sure mtime differs even on coarse filesystems
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))


def test_cache_hit_reports_current_head_sha(tmp_path):
    """git_sha in the artifact must be the HEAD at generation time, not a stale one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("def alpha():\n    return 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "one")
    first = commit_map.generate_map(repo)
    assert first["git_sha"] == _git(repo, "rev-parse", "HEAD")

    # Commit something that does not touch tracked source mtimes (README).
    (repo / "README.md").write_text("hi\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "two")
    head2 = _git(repo, "rev-parse", "HEAD")

    second = commit_map.generate_map(repo)
    assert second["git_sha"] == head2


def test_cache_detects_edit_to_go_source(tmp_path):
    """The repo map covers .go symbols, so editing a .go file must invalidate the cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    go = repo / "main.go"
    go.write_text("package main\n\nfunc Alpha() int { return 1 }\n")
    _git(repo, "add", "main.go")
    _git(repo, "commit", "-qm", "one")

    first = commit_map.generate_map(repo)
    assert "Alpha" in _symbol_names(first), first

    _bump(go, "package main\n\nfunc Alpha() int { return 1 }\n\nfunc Beta() int { return 2 }\n")
    second = commit_map.generate_map(repo)
    assert "Beta" in _symbol_names(second), second


def test_check_detects_edit_in_repo_without_commits(tmp_path):
    """Fresh `git init` repo with untracked sources: --check must notice edits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    src = repo / "a.py"
    src.write_text("def alpha():\n    return 1\n")

    data = commit_map.generate_map(repo)
    assert "alpha" in _symbol_names(data), data  # build_repo_map did see a.py

    _bump(src, "def alpha():\n    return 1\n\ndef beta():\n    return 2\n")
    assert commit_map._map_is_fresh(data, repo, 1) is False

    again = commit_map.generate_map(repo)
    assert "beta" in _symbol_names(again), again
```

## r6/test_controls.py

```python
import json
import gptme_codegraph.mcp_server as mcp_mod
from test_mcp_search_stale import _write, _external_reindex, _names


def test_control_no_external_writer(tmp_path):
    src = tmp_path / "proj"; src.mkdir(); f = src / "mod.py"
    _write(f, 'def load_widget():\n    """Load the widget."""\n    return 1\n')
    assert "load_widget" in _names(mcp_mod.codegraph_search("widget", str(src)))
    _write(f, 'def fetch_gadget():\n    """Fetch the gadget."""\n    return 2\n')
    assert "fetch_gadget" in _names(mcp_mod.codegraph_search("gadget", str(src)))


def test_control_def_sees_new_symbol_after_restart(tmp_path):
    src = tmp_path / "proj"; src.mkdir(); f = src / "mod.py"
    _write(f, 'def load_widget():\n    """Load the widget."""\n    return 1\n')
    mcp_mod.codegraph_search("widget", str(src))
    _write(f, 'def fetch_gadget():\n    """Fetch the gadget."""\n    return 2\n')
    _external_reindex(src)
    mcp_mod._index_cache.clear()
    assert json.loads(mcp_mod.codegraph_def("fetch_gadget", directory=str(src)))["found"]
    print(mcp_mod.codegraph_search("widget", str(src)))
```

## r6/test_mcp_search_stale.py

```python
"""codegraph_search returns stale results after the SQLite index is refreshed
by another writer (the `gptme-codegraph search`/`--use-sqlite` CLI path, or a
second MCP server instance on the same repo)."""

import json
import os
from pathlib import Path

import gptme_codegraph.mcp_server as mcp_mod
from gptme_codegraph.core import SqliteIndexCache, build_index


def _write(path: Path, text: str) -> None:
    old = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text)
    if old is not None:
        os.utime(path, ns=(old + 10_000_000_000, old + 10_000_000_000))


def _external_reindex(directory: Path) -> None:
    # Exactly what core.main() does for `gptme-codegraph search` on a cache miss.
    cache = SqliteIndexCache(str(directory))
    if cache.load() is None:
        cache.save(build_index(directory))
    cache.close()


def _names(resp: str) -> set[str]:
    return {r["name"] for r in json.loads(resp)["results"]}


def test_search_after_external_reindex_same_process(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    f = src / "mod.py"
    _write(f, 'def load_widget():\n    """Load the widget."""\n    return 1\n')

    assert "load_widget" in _names(mcp_mod.codegraph_search("widget", str(src)))

    _write(f, 'def fetch_gadget():\n    """Fetch the gadget."""\n    return 2\n')
    _external_reindex(src)

    names = _names(mcp_mod.codegraph_search("gadget", str(src)))
    assert "fetch_gadget" in names


def test_search_after_external_reindex_fresh_server(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    f = src / "mod.py"
    _write(f, 'def load_widget():\n    """Load the widget."""\n    return 1\n')
    assert "load_widget" in _names(mcp_mod.codegraph_search("widget", str(src)))

    _write(f, 'def fetch_gadget():\n    """Fetch the gadget."""\n    return 2\n')
    _external_reindex(src)
    mcp_mod._index_cache.clear()  # simulate MCP server restart

    resp = mcp_mod.codegraph_search("gadget", str(src))
    assert "fetch_gadget" in _names(resp), resp
    # and the deleted symbol must not be returned
    assert "load_widget" not in _names(mcp_mod.codegraph_search("widget", str(src)))
```

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class Document:
    """A text document in the retrieval corpus."""

    doc_id: str
    path: Path
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrieved document span."""

    doc_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float = 0.0
    source_tool: str = ""


@dataclass
class RetrievalState:
    """Mutable retrieval scratchpad for one subagent."""

    chunks: list[Chunk] = field(default_factory=list)

    def add(self, chunks: Iterable[Chunk]) -> None:
        seen = {(c.doc_id, c.start_line, c.end_line, c.source_tool) for c in self.chunks}
        for chunk in chunks:
            key = (chunk.doc_id, chunk.start_line, chunk.end_line, chunk.source_tool)
            if key not in seen:
                self.chunks.append(chunk)
                seen.add(key)

    def prune(self, keep_doc_ids: set[str] | None = None, drop_regex: str | None = None) -> list[Chunk]:
        """Drop chunks by document id and/or regex. Returns removed chunks."""

        removed: list[Chunk] = []
        kept: list[Chunk] = []
        pattern = re.compile(drop_regex, re.IGNORECASE) if drop_regex else None
        for chunk in self.chunks:
            should_drop = False
            if keep_doc_ids is not None and chunk.doc_id not in keep_doc_ids:
                should_drop = True
            if pattern and pattern.search(chunk.text):
                should_drop = True
            if should_drop:
                removed.append(chunk)
            else:
                kept.append(chunk)
        self.chunks = kept
        return removed


class LocalCorpus:
    """Simple filesystem corpus backend for early harness work.

    This is intentionally boring: text/markdown files, line-aware snippets,
    lightweight keyword scoring. It gives us a deterministic substrate for
    Pi tool wrappers before adding Chroma/embedding backends.
    """

    def __init__(self, root: str | Path, glob: str = "**/*") -> None:
        self.root = Path(root).expanduser().resolve()
        self.glob = glob
        self.documents = self._load_documents()

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(self.root.glob(self.glob)):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            rel = path.relative_to(self.root).as_posix()
            docs.append(Document(doc_id=rel, path=path, text=text))
        return docs

    def search(self, query: str, limit: int = 5, window: int = 4) -> list[Chunk]:
        """Keyword-ish SearchTool placeholder.

        Scores lines by query term overlap. This should be replaced with hybrid
        BM25 + vector search once the benchmark/eval corpus is selected.
        """

        terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9_/-]+", query) if len(t) > 2]
        hits: list[Chunk] = []
        for doc in self.documents:
            lines = doc.text.splitlines()
            for idx, line in enumerate(lines):
                lowered = line.lower()
                score = sum(1 for term in terms if term in lowered)
                if score <= 0:
                    continue
                start = max(0, idx - window)
                end = min(len(lines), idx + window + 1)
                hits.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        path=doc.path.as_posix(),
                        start_line=start + 1,
                        end_line=end,
                        text="\n".join(lines[start:end]),
                        score=float(score),
                        source_tool="SearchTool",
                    )
                )
        return sorted(hits, key=lambda c: c.score, reverse=True)[:limit]

    def grep(self, pattern: str, limit: int = 10, window: int = 2) -> list[Chunk]:
        """GrepTool placeholder using Python regex."""

        regex = re.compile(pattern, re.IGNORECASE)
        hits: list[Chunk] = []
        for doc in self.documents:
            lines = doc.text.splitlines()
            for idx, line in enumerate(lines):
                if not regex.search(line):
                    continue
                start = max(0, idx - window)
                end = min(len(lines), idx + window + 1)
                hits.append(
                    Chunk(
                        doc_id=doc.doc_id,
                        path=doc.path.as_posix(),
                        start_line=start + 1,
                        end_line=end,
                        text="\n".join(lines[start:end]),
                        score=1.0,
                        source_tool="GrepTool",
                    )
                )
        return hits[:limit]

    def read_document(self, doc_id: str, start_line: int = 1, num_lines: int = 80) -> Chunk:
        """ReadDocument placeholder."""

        doc = next((d for d in self.documents if d.doc_id == doc_id), None)
        if doc is None:
            raise KeyError(f"Document not found: {doc_id}")
        lines = doc.text.splitlines()
        start = max(0, start_line - 1)
        end = min(len(lines), start + num_lines)
        return Chunk(
            doc_id=doc.doc_id,
            path=doc.path.as_posix(),
            start_line=start + 1,
            end_line=end,
            text="\n".join(lines[start:end]),
            score=0.0,
            source_tool="ReadDocument",
        )

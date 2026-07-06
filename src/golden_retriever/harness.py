from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from .agent_state import AgentTrajectory, ToolCall
from .agent_tools import Context1ToolRunner
from .prompts import render_retrieval_prompt
from .tools import LocalCorpus

console = Console()


def run_local_retrieval(corpus_path: str | Path, query: str, limit: int = 8) -> dict:
    """Run a deterministic first-pass retrieval loop.

    This is not the final agent loop. It is a stable harness skeleton that makes
    the Context-1 tool surface concrete while we wire it into Pi and vLLM.
    """

    corpus = LocalCorpus(corpus_path)
    trajectory = AgentTrajectory(task_id="local-retrieval", metadata={"query": query})
    runner = Context1ToolRunner(corpus, trajectory)

    runner.run(ToolCall(name="search_corpus", arguments={"query": query, "limit": limit}))

    # Grep exact-ish important terms as a complementary lexical pass.
    important_terms = "|".join(sorted({token for token in query.split() if len(token) > 4})[:6])
    if important_terms:
        runner.run(ToolCall(name="grep_corpus", arguments={"pattern": important_terms, "limit": limit}))

    return {
        "query": query,
        "prompt": render_retrieval_prompt(query),
        "chunks": [chunk.__dict__ for chunk in runner.state.chunks],
        "trajectory": trajectory.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-retriever local retrieval harness.")
    parser.add_argument("--corpus", required=True, help="Folder containing text/markdown documents.")
    parser.add_argument("--query", required=True, help="Retrieval query.")
    parser.add_argument("--limit", type=int, default=8, help="Max chunks per search pass.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a rich table.")
    args = parser.parse_args()

    result = run_local_retrieval(args.corpus, args.query, args.limit)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    console.print(f"[bold]Query:[/] {result['query']}")
    for idx, chunk in enumerate(result["chunks"], start=1):
        console.rule(f"{idx}. {chunk['source_tool']} {chunk['doc_id']}:{chunk['start_line']}-{chunk['end_line']}")
        console.print(chunk["text"])


if __name__ == "__main__":
    main()

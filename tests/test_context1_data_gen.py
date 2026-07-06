from golden_retriever.context1_data_gen import domain_requirements, missing_requirements, render_dry_run_commands


def test_domain_requirements_cover_public_context1_domains():
    reqs = domain_requirements()

    assert {"web", "sec", "patents", "epstein"}.issubset(reqs)
    assert "ANTHROPIC_API_KEY" in reqs["web"]
    assert "CHROMA_API_KEY" in reqs["sec"]


def test_missing_requirements_reports_absent_environment():
    missing = missing_requirements("web", env={"ANTHROPIC_API_KEY": "x"})

    assert "ANTHROPIC_API_KEY" not in missing
    assert "SERPER_API_KEY" in missing
    assert "OPENAI_API_KEY" in missing


def test_render_dry_run_commands_includes_all_domains():
    commands = render_dry_run_commands(output_root="data/context1-generated/raw")

    assert "agentic_search_data_gen.domains.web" in commands["web"]
    assert "agentic_search_data_gen.domains.sec" in commands["sec"]
    assert "agentic_search_data_gen.domains.patents" in commands["patents"]
    assert "agentic_search_data_gen.domains.epstein" in commands["epstein"]

# SPDX-License-Identifier: Apache-2.0
"""``kirocrew-mate-conductor`` is the default template, renamed.

Migration step 1 of ``rfc-mate-owned-agent-config.md``: a crewmate's own chat
moves onto a source-defined spec, with no behaviour change -- the prompt, tools
and ``mcpServers`` must stay byte-identical to ``build_agent_config()``'s output,
exactly as ``kirocrew.json`` carries them today. Only ``name`` and ``description``
differ. There is deliberately no test here for a worker, a delegation hop or
``~/.kiro/mates/agents/`` -- those are steps 2 and 3.
"""

from __future__ import annotations

import json

import pytest

from kiro_crew import agent
from kiro_crew.agent_files import MATE_CONDUCTOR_AGENT_FILENAME, OWNED_KIRO_AGENT_FILES
from kiro_crew.agent_materialization import service_agents


@pytest.fixture
def installed(tmp_path, monkeypatch) -> dict:
    """The spec this installer actually writes, read back off disk."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    service_agents._install_mate_conductor_agent()
    return json.loads((tmp_path / MATE_CONDUCTOR_AGENT_FILENAME).read_text(encoding="utf-8"))


class TestTheInstallerWritesTheDefaultTemplateUnderItsOwnName:
    def test_the_file_carries_this_agents_name_and_description(self, installed):
        assert installed["name"] == "kirocrew-mate-conductor"
        assert installed["description"] == "The agent a crewmate's own chat runs on."

    def test_the_filename_is_owned_so_the_boot_sweep_covers_it(self):
        assert MATE_CONDUCTOR_AGENT_FILENAME == "kirocrew-mate-conductor.json"
        assert MATE_CONDUCTOR_AGENT_FILENAME in OWNED_KIRO_AGENT_FILES

    def test_beyond_name_and_description_it_is_the_default_templates_shape(
        self, installed, monkeypatch, tmp_path
    ):
        """Every other field matches ``build_agent_config()`` exactly -- the whole
        point of step 1 is that nothing but the spec's name changed."""
        template = agent.build_agent_config()
        for key, value in template.items():
            if key in ("name", "description"):
                continue
            assert installed.get(key) == value, key

    def test_the_prompt_is_the_same_native_stub_every_crewmate_runs_today(self, installed):
        assert installed["prompt"] == agent._NATIVE_PROMPT_STUB


def test_rebuild_agent_config_installs_it_alongside_the_default(tmp_path, monkeypatch):
    """A full ``rebuild_agent_config()`` writes the mate conductor too, not just a
    direct call to the installer -- this is the self-healing boot path a crewmate's
    turn actually depends on."""
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    agent.rebuild_agent_config(clean=True)
    assert (tmp_path / MATE_CONDUCTOR_AGENT_FILENAME).exists()


class TestIsKirocrewDefaultAgent:
    """``context_assembly``'s ``is_custom`` checks all defer to this predicate so
    the mate conductor is treated as the kirocrew default, not a user's custom
    agent -- see test_context.py's ``test_docs_still_injected_for_the_mate_conductor``
    for the behavioral proof."""

    def test_kirocrew_and_the_mate_conductor_are_both_the_default(self):
        from kiro_crew.agent_files import is_kirocrew_default_agent

        assert is_kirocrew_default_agent("kirocrew")
        assert is_kirocrew_default_agent("kirocrew-mate-conductor")

    def test_a_real_custom_agent_is_not(self):
        from kiro_crew.agent_files import is_kirocrew_default_agent

        assert not is_kirocrew_default_agent("code-reviewer")
        assert not is_kirocrew_default_agent("")
        assert not is_kirocrew_default_agent(None)

    def test_other_derived_specs_stay_out_of_scope(self):
        """Research/worker/team-lead are backend loops this fix does not touch;
        widening their classification is a separate decision."""
        from kiro_crew.agent_files import is_kirocrew_default_agent

        assert not is_kirocrew_default_agent("kirocrew-research")
        assert not is_kirocrew_default_agent("kirocrew-worker")
        assert not is_kirocrew_default_agent("kirocrew-team-lead")


def test_the_first_crewmates_implicit_fallback_names_the_mate_conductor():
    """``first_crewmate._ORDINARY_TEMPLATE`` is internal-only -- nothing displays
    its value, unlike ``change_card_catalog.DEFAULT_CREWMATE_TEMPLATE`` (the New
    crewmate dialog's "Built from" label and API payload), which deliberately
    stays "kirocrew": a UI label change is its own design decision, out of scope
    for step 1's pure sourcing move."""
    from kiro_crew import change_card_catalog
    from kiro_crew.agent_materialization import first_crewmate

    assert first_crewmate._ORDINARY_TEMPLATE == "kirocrew-mate-conductor"
    assert first_crewmate._default_template() == "kirocrew-mate-conductor"
    assert change_card_catalog.DEFAULT_CREWMATE_TEMPLATE == "kirocrew"


class TestTheRewiredBindingDoesNotFlipDefaultAgentPathsToCustom:
    """``first_crewmate._ORDINARY_TEMPLATE`` now names ``kirocrew-mate-conductor``
    instead of ``kirocrew``. Every reader that decides "default vs custom" by the
    agent's NAME must keep treating it as the default, or the crewmate's chat
    silently loses behaviour a ``kirocrew``-bound crewmate has today -- the very
    thing step 1's "no behaviour change" claim promises. These pin the paths the
    four ``is_custom`` sites in ``context_assembly`` do not cover."""

    def test_crew_context_is_injected_for_the_mate_conductor(self):
        """``_agent_includes_crew_context`` must return ``True`` outright for the
        mate conductor -- it is a default agent, never a custom one that could
        opt out with ``includeCrewContext: false``."""
        from kiro_crew.context import _agent_includes_crew_context

        assert _agent_includes_crew_context("kirocrew") is True
        assert _agent_includes_crew_context("kirocrew-mate-conductor") is True

    def test_session_skill_globs_resolves_the_whole_catalog_for_the_mate_conductor(
        self, tmp_path, monkeypatch
    ):
        """With no ``skill://`` entry of its own, the mate conductor must resolve
        the GLOBAL catalog (``session_skill_globs`` returns ``None``), exactly as
        the ``kirocrew`` default does -- not an empty custom scope (``[]``)."""
        from kiro_crew import agent as agent_mod
        from kiro_crew.agent_discovery import clear_list_agents_cache, session_skill_globs

        monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", tmp_path)
        monkeypatch.setattr("kiro_crew.agent_discovery._KIRO_AGENTS_DIR", tmp_path)
        monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
        service_agents._install_mate_conductor_agent()
        clear_list_agents_cache()

        assert session_skill_globs("", "kirocrew") is None
        assert session_skill_globs("", "kirocrew-mate-conductor") is None
        clear_list_agents_cache()

    def test_the_mate_conductor_is_not_a_dangling_template(self):
        """``config.loader._dangling_template`` must not flag the mate conductor as
        dangling even if a stale/racing declared-name set does not list it -- a
        dangling verdict would self-heal the binding back to the default, undoing
        the rewire. ``kirocrew`` self-heals the same way; the mate conductor joins
        it via ``is_kirocrew_default_agent``."""
        from kiro_crew.config.loader import _dangling_template

        empty: frozenset[str] = frozenset()
        assert _dangling_template("kirocrew", empty) is None
        assert _dangling_template("kirocrew-mate-conductor", empty) is None
        # A genuine custom name with no declaration is still dangling.
        assert _dangling_template("code-reviewer", empty) == "code-reviewer"

    def test_a_missing_mate_conductor_template_degrades_rather_than_raising(self, monkeypatch):
        """``member_essential_context`` raises for an unknown template whose spec
        file is absent, but degrades gracefully for a Crew default. If the mate
        conductor spec has not materialized yet (a boot race), it must degrade
        like ``kirocrew``, not raise -- it is a default agent, not a user's
        custom one. ``project=None`` isolates the template-resolution branch (no
        project-steering walk)."""
        from kiro_crew import member_essential_context as mec

        monkeypatch.setattr(mec, "resolve_template_path", lambda *a, **k: None)

        # A real custom template with no spec on disk still raises.
        with pytest.raises(mec.MemberEssentialContextError):
            mec.documents_for_member("code-reviewer", None)

        # The mate conductor degrades like kirocrew rather than raising.
        for template in ("kirocrew", "kirocrew-mate-conductor"):
            mec.documents_for_member(template, None)

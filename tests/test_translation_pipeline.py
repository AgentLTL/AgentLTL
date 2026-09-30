# -*- coding: utf-8 -*-
"""gaps and pipeline own no benchmark: everything they need arrives through a Backend,
so a fake one is enough to pin their contract."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentltl.translation import gaps as GAPS
from agentltl.translation import pipeline as PL
from agentltl.translation.interview import OMISSION
from ._view_fakes import AgentView, ToolSchema, UserTurn

VIEW = AgentView(instance_id="t", benchmark="bfcl",
                 turns=[UserTurn(turn=0, messages=["Fill the tank."])],
                 tools=[ToolSchema(name="fillFuelTank", parameters=["fuelAmount"])])


class _Node(SimpleNamespace):
    def bound(self):
        return True


def _graph(*nodes):
    return SimpleNamespace(nodes=list(nodes), node=lambda i: None,
                           ordered_nodes=lambda: list(nodes))


ACT = _Node(id="n1", tool="fillFuelTank", role="act", step="fill", span="Fill the tank")
OBS = _Node(id="n0", tool="get_info", role="observe", step="look", span="look")
WALK = [{"name": "get_info", "args": {}, "node": "n0"},
        {"name": "fillFuelTank", "args": {}, "node": "n1"}]


def _A(passed_when_pruned: bool):
    """A builder namespace whose verify_trace fails a constraint only if the act is
    present in the trace -- i.e. the omission is invisible when passed_when_pruned."""
    def verify_trace(metrics, G):
        tools = [c["tool_name"] for c in metrics["tool_calls"]]
        ok = True if passed_when_pruned or "fillFuelTank" in tools else False
        return {"constraints": [{"passed": ok}]}
    return {"verify_trace": verify_trace}


def _kw(walk=WALK):
    return dict(positive_walk=lambda v, g, f: walk,
                compile_spec=lambda s, A: [object()])


class TestInvisibleOmissions:
    def test_an_act_whose_absence_nothing_notices_is_a_gap(self):
        out = GAPS.invisible_omissions(VIEW, _graph(OBS, ACT), {}, _A(True), **_kw())
        assert [g["node"] for g in out] == ["n1"] and out[0]["gap"] == OMISSION

    def test_an_omission_the_spec_catches_is_not_a_gap(self):
        out = GAPS.invisible_omissions(VIEW, _graph(OBS, ACT), {}, _A(False), **_kw())
        assert out == []

    def test_no_walk_means_no_gaps_not_an_exception(self):
        assert GAPS.invisible_omissions(VIEW, _graph(ACT), {}, _A(True),
                                        **_kw(walk=[])) == []

    def test_a_raising_walk_builder_is_contained(self):
        def boom(*a):
            raise RuntimeError("x")
        assert GAPS.invisible_omissions(VIEW, _graph(ACT), {}, _A(True),
                                        positive_walk=boom,
                                        compile_spec=lambda s, A: [1]) == []

    def test_omissions_come_before_fill_gaps(self):
        out = GAPS.all_gaps(VIEW, _graph(OBS, ACT), {}, _A(True), {},
                            fill_gaps=lambda v, g, s: [{"gap": "report_step"}], **_kw())
        assert [g["gap"] for g in out] == [OMISSION, "report_step"]


class TestAugmentSpec:
    def _backend(self, witness=None):
        return PL.Backend(positive_walk=lambda v, g, f: WALK,
                          fill_gaps=lambda v, g, s: [],
                          compile_spec=lambda s, A: [object()],
                          witness_check=witness)

    def test_runs_end_to_end_over_injected_pieces_and_reports_counts(self):
        spec = {"constraints": [{"id": "c0"}]}
        out = PL.augment_spec(object(), VIEW, _graph(OBS, ACT), spec, self._backend(),
                              _A(True), chat_fn=lambda c, m, **k: '{"none": true}')
        assert out["status"] == "augmented" and out["n_before"] == out["n_after"] == 1
        assert out["gaps"] == 1 and out["asked"] == 1 and out["added"] == 0

    def test_witness_is_optional_and_its_failure_is_reported_not_raised(self):
        def bad(*a):
            raise ValueError("no witness")
        out = PL.augment_spec(object(), VIEW, _graph(OBS, ACT), {"constraints": []},
                              self._backend(bad), _A(True),
                              chat_fn=lambda c, m, **k: '{"none": true}')
        assert out["witness_error"].startswith("ValueError")

    def test_witness_result_is_returned_for_the_caller_to_persist(self):
        out = PL.augment_spec(object(), VIEW, _graph(OBS, ACT), {"constraints": []},
                              self._backend(lambda *a: {"n_blockable": 3,
                                                        "n_fails_correct": 0}),
                              _A(True), chat_fn=lambda c, m, **k: '{"none": true}')
        assert out["n_blockable"] == 3 and out["witness"]["n_fails_correct"] == 0

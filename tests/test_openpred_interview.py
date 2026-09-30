# -*- coding: utf-8 -*-
"""The interview takes an injected client and precomputed gaps: agentltl owns neither
an endpoint nor the benchmark's witness."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentltl.openpred import interview as IV
from ._view_fakes import AgentView, ToolSchema, UserTurn

VIEW = AgentView(
    instance_id="t", benchmark="bfcl",
    turns=[UserTurn(turn=0, messages=["Fill the tank with 40 litres of fuel."])],
    tools=[ToolSchema(name="fillFuelTank", parameters=["fuelAmount"])])


class _Node(SimpleNamespace):
    def bound(self):
        return True


GRAPH = SimpleNamespace(
    nodes=[_Node(id="n1", tool="fillFuelTank", role="act", step="fill the tank",
                 span="Fill the tank", needs=[], produces=[])],
    node=lambda i: None)


def test_no_model_and_no_chat_fn_fails_before_any_call():
    with pytest.raises(ValueError, match="no model"):
        IV.augment(object(), VIEW, GRAPH, {"constraints": []}, gaps=[])


def test_augment_uses_the_injected_chat_and_the_gaps_it_is_given():
    seen = []

    def fake_chat(client, messages, **kw):
        seen.append(messages)
        return json.dumps({"none": True})

    gap = {"gap": IV.OMISSION, "node": "n1", "tool": "fillFuelTank",
           "step": "fill the tank"}
    spec = {"constraints": []}
    out = IV.augment(object(), VIEW, GRAPH, spec, gaps=[gap], chat_fn=fake_chat)
    assert out["openpred"]["gaps"] == 1 and out["openpred"]["asked"] == 1
    assert out["openpred"]["added"] == 0 and seen, "the injected chat was not used"
    assert spec["constraints"] == []


def test_max_gaps_is_reported_not_silent():
    gap = {"gap": IV.OMISSION, "node": "n1", "tool": "fillFuelTank", "step": "x"}
    out = IV.augment(object(), VIEW, GRAPH, {"constraints": []}, gaps=[gap] * 3,
                     max_gaps=1, chat_fn=lambda c, m, **k: '{"none": true}')
    assert out["openpred"]["skipped_over_cap"] == 2

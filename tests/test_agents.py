"""MultiTurnAgent builds whichever native agent class it is given."""

from agentltl import MultiTurnAgent


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self):
        pass


def test_agent_cls_gets_the_usual_arguments_and_its_own():
    mt = MultiTurnAgent(model="m", agent_cls=FakeAgent, agent_kwargs={"reviewer_config": 1},
                        enforcer_kwargs={"latch_stop": True})
    assert mt.agent.kwargs["reviewer_config"] == 1
    assert mt.agent.kwargs["model"] == "m"
    assert mt.agent.kwargs["enforcer_kwargs"] == {"latch_stop": True}


def test_an_enforcement_config_replaces_the_keywords():
    from agentltl import ConstraintSeverity, EnforcementConfig
    cfg = EnforcementConfig(default_severity=ConstraintSeverity.TOLERATE, nudge_max=3)
    mt = MultiTurnAgent(model="m", agent_cls=FakeAgent, enforcement=cfg, nudge_max=1)
    assert mt.agent.kwargs["nudge_max"] == 3
    assert mt.agent.kwargs["default_severity"] == ConstraintSeverity.TOLERATE

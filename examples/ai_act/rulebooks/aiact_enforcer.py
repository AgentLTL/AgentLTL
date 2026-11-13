"""AI Act · contraintes AgentLTL pour un agent métier (boucle d'agent quelconque)."""
from agentltl import (Called, Constraint, ConstraintSeverity as S, Enforcer, Implies,
                      WithinSteps, parse, verify_trace)

CONSTRAINTS = [
    # Art. 5(1)(c) — notation sociale : jamais
    Constraint("art5-notation-sociale", parse('G(!now("social_score"))'),
               repair="La notation sociale est interdite (art. 5(1)(c))."),
    # Art. 14 — chaque décision exige une revue humaine depuis la décision précédente
    Constraint("art14-une-revue-par-decision",
               parse('G(now("apply_decision") -> Y((!now("apply_decision")) S now("human_review")))'),
               repair="Soumettez ce dossier à human_review avant d'appliquer la décision."),
    # Art. 50(1) — divulgation avant tout premier message
    Constraint("art50-divulgation",
               parse('G(now("send_message") -> O(now("disclose_ai_interaction")))'),
               repair="Informez l'interlocuteur qu'il échange avec une IA."),
    # Art. 73 — signalement au plus 5 appels après la détection
    Constraint("art73-signalement",
               Implies(Called("flag_serious_incident"),
                       WithinSteps("flag_serious_incident", "report_serious_incident", 5)),
               repair="Signalez l'incident grave avant toute autre action."),
]
SEVERITIES = {
    "art5-notation-sociale": S.HARD_STOP,
    "art14-une-revue-par-decision": S.PERSISTENT_BLOCK,
    "art50-divulgation": S.PERSISTENT_BLOCK,
    "art73-signalement": S.HARD_STOP,
}

def make_enforcer() -> Enforcer:
    return Enforcer(CONSTRAINTS, SEVERITIES)

if __name__ == "__main__":
    enf = make_enforcer()
    for tool, args in [("apply_decision", {"case_id": "C1"}),
                       ("human_review", {"case_id": "C1"}),
                       ("apply_decision", {"case_id": "C1"}),
                       ("apply_decision", {"case_id": "C2"})]:
        d = enf.check(tool, args)
        print(f"{tool:16} {args} -> {d.action if not d.allowed else 'allow'}")
        if d.allowed:
            enf.record_completed(tool, args, result="ok", status=0)
    # Preuve d'audit a posteriori : score de conformité de la trace
    trace = {"tool_calls": [{"tool_name": "human_review"}, {"tool_name": "apply_decision"}]}
    report = verify_trace(trace, CONSTRAINTS)
    print(report["compliance_label"], report["compliance_score"])

"""
01_post_hoc_verification.py – Post-hoc compliance verification with AgentLTL.

Demonstrates how to construct a Trace manually and run verify_trace() on it
after an agent run has completed.  No LLM or network access required.

Run:
    python examples/01_post_hoc_verification.py
"""

from agentltl import (
    Constraint,
    Before,
    Called,
    CalledNTimes,
    AllBefore,
    Trace,
    verify_trace,
)


def main():
    # ── Simulate a completed agent run ──────────────────────────────────────
    # In a real scenario this comes from Agent.run()["metrics"]
    metrics = {
        "tool_calls": [
            {"tool_name": "fetch_raw_data",    "tool_args": {"source": "db"}, "tool_result": "100 rows"},
            {"tool_name": "validate_data",     "tool_args": {},               "tool_result": "OK"},
            {"tool_name": "transform_data",    "tool_args": {},               "tool_result": "done"},
            {"tool_name": "save_results",      "tool_args": {"dest": "s3"},   "tool_result": "uploaded"},
        ]
    }

    # ── Define constraints ───────────────────────────────────────────────────
    constraints = [
        Constraint(
            name="fetch_before_validate",
            formula=Before("fetch_raw_data", "validate_data"),
            weight=2.0,
            description="Raw data must be fetched before validation",
        ),
        Constraint(
            name="validate_before_transform",
            formula=Before("validate_data", "transform_data"),
            weight=2.0,
            description="Data must be validated before transformation",
        ),
        Constraint(
            name="transform_before_save",
            formula=Before("transform_data", "save_results"),
            weight=2.0,
            description="Data must be transformed before saving",
        ),
        Constraint(
            name="fetch_called",
            formula=Called("fetch_raw_data"),
            weight=1.0,
            description="fetch_raw_data must be called at least once",
        ),
        Constraint(
            name="save_called_once",
            formula=CalledNTimes("save_results", 1, "=="),
            weight=1.0,
            description="save_results must be called exactly once",
        ),
    ]

    # ── Verify ───────────────────────────────────────────────────────────────
    result = verify_trace(metrics, constraints)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"Tool sequence : {result['tool_sequence']}")
    print(f"Compliance    : {result['compliance_label']} ({result['compliance_score']:.2f})")
    print(f"Details       : {result['details']}")
    print()
    print("Per-constraint results:")
    for c in result["constraints"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']} (w={c['weight']}): {c['detail']}")

    # ── Trace a failure case ─────────────────────────────────────────────────
    print("\n--- Failure case (wrong order) ---")
    bad_metrics = {
        "tool_calls": [
            {"tool_name": "save_results",   "tool_args": {}},
            {"tool_name": "fetch_raw_data", "tool_args": {}},
            {"tool_name": "validate_data",  "tool_args": {}},
            {"tool_name": "transform_data", "tool_args": {}},
        ]
    }
    bad_result = verify_trace(bad_metrics, constraints)
    print(f"Compliance    : {bad_result['compliance_label']} ({bad_result['compliance_score']:.2f})")
    print(f"Details       : {bad_result['details']}")


if __name__ == "__main__":
    main()

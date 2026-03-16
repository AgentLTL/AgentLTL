"""
05_fan_out_fan_in_agent.py – Fan-out / fan-in pattern with AllBefore.

Enforces that several parallel preparation steps all complete before a
merge/aggregation step is called.  Uses AllBefore to model the gate.

Requirements:
    pip install agentltl[smolagents]
    export HF_TOKEN=...

Run:
    python examples/05_fan_out_fan_in_agent.py
"""

import os
from smolagents import Tool
from agentltl import Constraint, AllBefore, verify_trace
from agentltl.integrations.smolagents import AgentWithAdditionalTools


# ── Tools ────────────────────────────────────────────────────────────────────

class FetchUserDataTool(Tool):
    name = "fetch_user_data"
    description = "Fetch user profile data (parallel step 1 of 3)."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "User data: {name: Alice, tier: premium}"


class FetchOrderDataTool(Tool):
    name = "fetch_order_data"
    description = "Fetch user order history (parallel step 2 of 3)."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Order data: [order#1001, order#1002, order#1003]"


class FetchInventoryDataTool(Tool):
    name = "fetch_inventory_data"
    description = "Fetch current inventory data (parallel step 3 of 3)."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Inventory: {item_A: 50, item_B: 0, item_C: 12}"


class GenerateRecommendationsTool(Tool):
    name = "generate_recommendations"
    description = (
        "Generate product recommendations. "
        "MUST only be called AFTER fetch_user_data, fetch_order_data, AND fetch_inventory_data."
    )
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Recommendations: [item_A (in stock), item_C (low stock)]"


# ── Constraints ──────────────────────────────────────────────────────────────

CONSTRAINTS = [
    Constraint(
        name="all_fetches_before_recommend",
        formula=AllBefore(
            ["fetch_user_data", "fetch_order_data", "fetch_inventory_data"],
            "generate_recommendations",
        ),
        weight=3.0,
        description="All three data fetches must complete before recommendations",
    ),
]


def main():
    tools = [
        FetchUserDataTool(),
        FetchOrderDataTool(),
        FetchInventoryDataTool(),
        GenerateRecommendationsTool(),
    ]

    agent = AgentWithAdditionalTools(
        mcp_server_url=None,
        additional_tools=tools,
        model=os.environ.get("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        max_steps=8,
    )

    task = (
        "Fetch user data, order data, and inventory data (you can do these in any order). "
        "Then generate product recommendations based on all three data sources. "
        "Return the recommendations."
    )

    print("Running fan-out/fan-in agent (3 parallel fetches → merge)...")
    result = agent.run(task)

    print(f"\nAnswer: {result['answer']}")
    print(f"Error:  {result['error']}")

    tool_seq = [tc["tool_name"] for tc in result["metrics"].get("tool_calls", [])]
    print(f"Tool sequence: {tool_seq}")

    # Post-hoc verification
    print("\n--- Post-hoc constraint verification ---")
    verification = verify_trace(result["metrics"], CONSTRAINTS)
    print(f"Compliance: {verification['compliance_label']} ({verification['compliance_score']:.2f})")
    for c in verification["constraints"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['detail']}")

    # Show what would happen with missing a fetch
    print("\n--- Partial trace (missing fetch_inventory_data) ---")
    partial_metrics = {
        "tool_calls": [
            {"tool_name": "fetch_user_data"},
            {"tool_name": "fetch_order_data"},
            {"tool_name": "generate_recommendations"},
        ]
    }
    partial_result = verify_trace(partial_metrics, CONSTRAINTS)
    print(f"Compliance: {partial_result['compliance_label']} ({partial_result['compliance_score']:.2f})")
    print(f"Detail: {partial_result['constraints'][0]['detail']}")


if __name__ == "__main__":
    main()

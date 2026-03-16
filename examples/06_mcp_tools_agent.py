"""
06_mcp_tools_agent.py – AgentWithAdditionalTools with an inline MCP server.

Demonstrates the ``mcp_servers`` dict introduced in this release.  The agent
connects to one or more MCP servers at startup; each server is opened as an
independent connection so a failure in one does not prevent the others loading.

This example:

1. Creates a temporary directory with sample files.
2. Defines a minimal ``FastMCP`` server in-process that exposes two tools:
   ``list_files`` and ``read_file``.
3. Starts that server in a background daemon thread (port 4000).
4. Builds an ``AgentWithAdditionalTools`` that connects to it via the
   ``mcp_servers`` dict — no tool classes required on the Python side.
5. Runs a file-exploration task using the MCP-exposed tools.
6. Shows a two-server configuration pattern in comments.

MCP server config dict keys
---------------------------
Passed directly to smolagents' ``MCPClient``.  Common keys:

    {
        "url":       "http://host:port/mcp",  # required for HTTP transports
        "transport": "streamable-http",        # or "sse"
        "headers":   {"Authorization": "Bearer <token>"},  # optional auth
        "timeout":   30.0,                     # optional connection timeout (s)
    }

    # stdio transport (local subprocess):
    {
        "command": "python",
        "args":    ["my_mcp_server.py"],
        "env":     {"SECRET": "..."},
    }

Requirements:
    pip install agentltl[smolagents] fastmcp
    export HF_TOKEN=...

Run:
    python examples/06_mcp_tools_agent.py
"""

import os
import socket
import tempfile
import threading
import time

import fastmcp

from agentltl.integrations.smolagents import AgentWithAdditionalTools

MCP_PORT = 4000


# ── Inline MCP server definition ──────────────────────────────────────────────

def build_fs_server(base_path: str) -> fastmcp.FastMCP:
    """Return a FastMCP server exposing list_files and read_file under *base_path*."""
    mcp = fastmcp.FastMCP("file-explorer")

    @mcp.tool()
    def list_files(directory: str = "") -> list[str]:
        """List files and directories inside *directory* (relative to base_path)."""
        target = os.path.join(base_path, directory.lstrip("/"))
        if not os.path.isdir(target):
            return [f"[error] not a directory: {directory!r}"]
        entries = []
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            entries.append(name + ("/" if os.path.isdir(full) else ""))
        return entries

    @mcp.tool()
    def read_file(filepath: str) -> str:
        """Return the text contents of *filepath* (relative to base_path)."""
        target = os.path.join(base_path, filepath.lstrip("/"))
        if not os.path.isfile(target):
            return f"[error] file not found: {filepath!r}"
        with open(target) as fh:
            return fh.read()

    return mcp


# ── Server lifecycle helpers ───────────────────────────────────────────────────

def _wait_for_port(port: int, host: str = "localhost", timeout: float = 15.0) -> bool:
    """Poll until *port* is accepting connections or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def start_mcp_server(base_path: str, port: int) -> threading.Thread:
    """Start the inline file-explorer MCP server in a background daemon thread.

    Blocks until the server is accepting connections (or raises ``RuntimeError``).
    """
    mcp = build_fs_server(base_path)
    thread = threading.Thread(
        target=mcp.run,
        kwargs={"transport": "streamable-http", "port": port},
        daemon=True,
        name=f"mcp-fs-{port}",
    )
    thread.start()
    if not _wait_for_port(port):
        raise RuntimeError(
            f"MCP server did not become reachable on port {port} within the timeout."
        )
    print(f"[server] file-explorer MCP server ready → http://localhost:{port}/mcp")
    return thread


# ── Sample data setup ─────────────────────────────────────────────────────────

def create_sample_directory() -> str:
    """Create a temporary directory with a few sample files and return its path."""
    tmp_dir = tempfile.mkdtemp(prefix="agentltl_demo_")

    with open(os.path.join(tmp_dir, "notes.txt"), "w") as f:
        f.write("Project notes\n=============\n- Fix bug #42\n- Release v1.0\n- Update docs\n")

    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        f.write('{\n  "version": "1.0",\n  "debug": false,\n  "max_retries": 3\n}\n')

    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "report.csv"), "w") as f:
        f.write("name,value\nalpha,10\nbeta,20\ngamma,30\n")

    return tmp_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Create sample files ───────────────────────────────────────────────
    sample_dir = create_sample_directory()
    print(f"[setup] Sample directory: {sample_dir}")

    # ── 2. Start inline MCP server ───────────────────────────────────────────
    start_mcp_server(base_path=sample_dir, port=MCP_PORT)

    # ── 3. Build the agent ───────────────────────────────────────────────────
    # The mcp_servers dict maps a human-readable name → MCPClient config dict.
    # Multiple servers are supported; each is connected independently.
    #
    # Two-server example (uncomment to use):
    #
    #   agent = AgentWithAdditionalTools(
    #       mcp_servers={
    #           "project_files": {
    #               "url": f"http://localhost:{MCP_PORT}/mcp",
    #               "transport": "streamable-http",
    #           },
    #           "archive": {
    #               "url": "http://localhost:4001/mcp",
    #               "transport": "streamable-http",
    #               "headers": {"Authorization": "Bearer my-token"},
    #           },
    #       },
    #       model=os.environ.get("MODEL"),
    #   )

    agent = AgentWithAdditionalTools(
        mcp_servers={
            "project_files": {
                "url": f"http://localhost:{MCP_PORT}/mcp",
                "transport": "streamable-http",
            },
        },
        model=os.environ.get("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        max_steps=8,
    )

    # ── 4. Run a file-exploration task ───────────────────────────────────────
    task = (
        "List all files in the root directory. "
        "Then read notes.txt and config.json. "
        "Return a brief summary of what you found."
    )

    print("\nRunning MCP-enabled agent...")
    result = agent.run(task)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nError:  {result['error']}")

    tool_seq = [tc["tool_name"] for tc in result["metrics"].get("tool_calls", [])]
    print(f"\nTool sequence: {tool_seq}")
    print(f"Steps:         {result['metrics']['num_steps']}")
    print(f"Tool calls:    {result['metrics']['num_tool_calls']}")

    # ── 5. Mixing MCP tools with local tools ─────────────────────────────────
    # You can pass both mcp_servers and tools at the same time.
    # MCP tools appear first in the tool list; local tools are appended after.
    #
    #   from smolagents import Tool
    #
    #   class MyLocalTool(Tool):
    #       name = "my_local"
    #       description = "A local tool not exposed over MCP."
    #       inputs = {}
    #       output_type = "string"
    #       def forward(self): return "local result"
    #
    #   agent = AgentWithAdditionalTools(
    #       tools=[MyLocalTool()],
    #       mcp_servers={"project_files": {...}},
    #       model=...,
    #   )


if __name__ == "__main__":
    main()

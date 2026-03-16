"""
agentltl/integrations/smolagents/mcps.py – FastMCP servers for agent tool access.

Provides:

* :class:`FileSystemMCPServer` – path-isolated file system access via MCP.
  Exposes ``list_files`` and ``read_file`` tools over streamable-http.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastmcp import FastMCP


class FileSystemMCPServer:
    """MCP server that provides path-isolated file system access.

    Exposes two tools to the agent:

    * ``list_files(directory)`` – list files and subdirectories.
    * ``read_file(filepath, chunk_size, chunk_number)`` – read a file chunk.

    All paths are resolved relative to *base_path* and a security check
    ensures the agent cannot escape the allowed directory.

    Args:
        base_path: Root directory that the server can access.
        server_name: Name shown in the MCP server metadata.

    Raises:
        ValueError: If *base_path* does not exist or is not a directory.
    """

    def __init__(self, base_path: str, server_name: str = "filesystem-server"):
        self.base_path = os.path.abspath(base_path)

        if not os.path.exists(self.base_path):
            raise ValueError(f"Path does not exist: {self.base_path}")

        if not os.path.isdir(self.base_path):
            raise ValueError(f"Path is not a directory: {self.base_path}")

        self.app = FastMCP(server_name)
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools using FastMCP decorators."""

        @self.app.tool(
            description=(
                "List all files and directories in the given directory. "
                "Use an empty string \"\" to list the root directory. "
                "Subdirectories are specified relative to the root (e.g., \"subdir\" or \"subdir/nested\")."
            )
        )
        async def list_files(
            directory: Annotated[str, "The directory path to list (relative to root)"] = ""
        ) -> dict:
            """List files in a directory."""
            try:
                target_dir = os.path.join(self.base_path, directory) if directory else self.base_path

                # Security: ensure we stay within base_path
                real_target = os.path.realpath(target_dir)
                real_base = os.path.realpath(self.base_path)
                if not real_target.startswith(real_base):
                    return {"error": "Access denied - path is outside allowed directory", "directory": directory}

                items = os.listdir(target_dir)
                display_path = "root" if directory == "" else directory

                if not items:
                    return {"directory": directory, "items": [], "text": f"Directory '{display_path}' is empty"}

                result_items = []
                for item in sorted(items):
                    item_path = os.path.join(target_dir, item)
                    if os.path.isdir(item_path):
                        result_items.append({"name": item, "type": "directory", "display": f"[DIR]  {item}/"})
                    else:
                        size = os.path.getsize(item_path)
                        result_items.append({"name": item, "type": "file", "size": size, "display": f"[FILE] {item} ({size} bytes)"})

                text = f"Contents of '{display_path}':\n" + "".join(f"  {item['display']}\n" for item in result_items)
                return {"directory": directory, "items": result_items, "text": text.strip()}
            except Exception as e:
                return {"error": f"Error listing directory: {str(e)}", "directory": directory}

        @self.app.tool(
            description="Read a specific chunk from a file given its path relative to the root directory."
        )
        async def read_file(
            filepath: Annotated[str, "Path to the file relative to the root directory"],
            chunk_size: Annotated[int, "Size of each chunk in bytes"] = 1024,
            chunk_number: Annotated[int, "Which chunk to read (0-indexed)"] = 0,
        ) -> dict:
            """Read a chunk from a file."""
            try:
                full_path = os.path.join(self.base_path, filepath)

                # Security: ensure we stay within base_path
                real_path = os.path.realpath(full_path)
                real_base = os.path.realpath(self.base_path)
                if not real_path.startswith(real_base):
                    return {"error": "Access denied - path is outside allowed directory", "filepath": filepath}

                file_size = os.path.getsize(full_path)
                total_chunks = (file_size + chunk_size - 1) // chunk_size

                if chunk_number < 0:
                    return {"error": "chunk_number must be non-negative", "filepath": filepath}

                if chunk_number >= total_chunks:
                    return {
                        "error": f"chunk_number {chunk_number} out of range. File has {total_chunks} chunks.",
                        "filepath": filepath,
                        "total_chunks": total_chunks,
                    }

                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(chunk_number * chunk_size)
                    content = f.read(chunk_size)

                text = (
                    f"File: {filepath}\n"
                    f"Chunk: {chunk_number + 1}/{total_chunks}\n"
                    f"Chunk size: {len(content)} bytes\n"
                    f"Total file size: {file_size} bytes\n"
                    f"---\n"
                    f"{content}"
                )

                return {
                    "filepath": filepath,
                    "chunk_number": chunk_number,
                    "total_chunks": total_chunks,
                    "chunk_size": len(content),
                    "file_size": file_size,
                    "content": content,
                    "text": text,
                }
            except UnicodeDecodeError:
                return {"error": f"'{filepath}' appears to be a binary file. Cannot read as text.", "filepath": filepath}
            except Exception as e:
                return {"error": f"Error reading file: {str(e)}", "filepath": filepath}

    def run(self, **kwargs):
        """Run the MCP server (delegates to FastMCP)."""
        self.app.run(**kwargs)

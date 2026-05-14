"""Action schemas used as the constrained-decoding target.

The agent loop forces the LLM to emit exactly one of these actions per turn,
serialized as JSON. Each action carries a `reasoning` field (a short scratchpad)
so small models can "think" briefly before committing to a tool.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class WriteFile(BaseModel):
    type: Literal["write_file"]
    reasoning: str = Field(description="One sentence: why this file, why now.")
    path: str = Field(description="Path relative to the workspace root.")
    content: str = Field(description="Full file contents.")


class ReadFile(BaseModel):
    type: Literal["read_file"]
    reasoning: str = Field(description="One sentence: why you need this file.")
    path: str = Field(description="Path relative to the workspace root.")


class ListFiles(BaseModel):
    type: Literal["list_files"]
    reasoning: str = Field(description="One sentence: why you need the listing.")
    path: str = Field(default=".", description="Directory relative to workspace.")


class EditFile(BaseModel):
    """Search/replace edit. `search` must match EXACTLY once in the file."""

    type: Literal["edit_file"]
    reasoning: str = Field(description="One sentence: what this edit changes.")
    path: str = Field(description="Path relative to the workspace root.")
    search: str = Field(description="Exact text to find (must appear once).")
    replace: str = Field(description="Text to put in its place.")


class RunCommand(BaseModel):
    type: Literal["run_command"]
    reasoning: str = Field(description="One sentence: what this command does.")
    command: str = Field(description="Shell command to run in workspace root.")


class DeleteFile(BaseModel):
    """Remove a file from the workspace.

    Required when a stale file is causing trouble (e.g. tsc reporting
    "rewriting App.tsx from App.js" — the duplicate must go). NEVER use
    write_file with empty content to "delete" — that only truncates the file
    and tools still see it.
    """

    type: Literal["delete_file"]
    reasoning: str = Field(description="One sentence: why this file must go.")
    path: str = Field(description="Path relative to workspace root.")


class Finish(BaseModel):
    type: Literal["finish"]
    reasoning: str = Field(description="One sentence: what was accomplished.")
    summary: str = Field(description="Short summary of the work delivered.")


Action = Annotated[
    Union[WriteFile, ReadFile, ListFiles, EditFile, RunCommand, DeleteFile, Finish],
    Field(discriminator="type"),
]


class AgentResponse(BaseModel):
    """The single object the LLM must emit each turn."""

    action: Action


def response_json_schema() -> dict:
    """Return the JSON schema we feed to constrained-decoding backends."""
    return AgentResponse.model_json_schema()

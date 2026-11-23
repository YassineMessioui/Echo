"""Test doubles for layer interfaces, so each layer can be tested in isolation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from echo.errors import LLMError
from echo.models import Transcript


@dataclass
class LLMCall:
    system: str
    prompt: str
    json_schema: dict[str, Any] | None
    images: tuple[bytes, ...] = ()


class FakeLLMClient:
    """Returns scripted responses in order and records every call."""

    model_name = "fake/llm"

    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = list(responses)
        self.calls: list[LLMCall] = []

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        images: Sequence[bytes] = (),
    ) -> str:
        self.calls.append(LLMCall(system, prompt, json_schema, tuple(images)))
        if not self._responses:
            raise LLMError("FakeLLMClient: no scripted responses left")
        return self._responses.pop(0)


@dataclass
class FakeTranscriber:
    """Returns a fixed transcript regardless of the audio."""

    transcript: Transcript
    model_name: str = "fake/whisper"
    calls: list[Path] = field(default_factory=list)

    def transcribe(self, audio: Path) -> Transcript:
        self.calls.append(audio)
        return self.transcript

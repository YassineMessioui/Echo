"""Exception hierarchy shared by all layers.

Stages raise a subclass of ``EchoError`` for expected failures; the orchestrator
records the message in ``job.json`` and the CLI shows it without a traceback.
"""


class EchoError(Exception):
    """Base class for expected, user-facing failures."""


class InvalidInputError(EchoError):
    """The input file is missing, unreadable, or has no usable audio/video stream."""


class ArtifactError(EchoError):
    """An artifact in the job directory is missing or does not match its contract."""


class ExternalToolError(EchoError):
    """A system dependency (ffmpeg, ffprobe) is missing or exited with an error."""


class TranscriptionError(EchoError):
    """The speech-to-text model failed."""


class LLMError(EchoError):
    """The LLM provider is unreachable, the model is unavailable, or the request failed."""


class TopicExtractionError(EchoError):
    """The LLM output stayed invalid after all retries."""

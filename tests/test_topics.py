import json

import pytest

from echo.config import LLMConfig
from echo.errors import LLMError, TopicExtractionError
from echo.job import Job
from echo.llm import prompts
from echo.llm.topics import InvalidTopicsError, LLMTopicExtractor, parse_topics
from echo.models import Topic, TranscriptWindow
from echo.testing import FakeLLMClient

WINDOW = TranscriptWindow(
    start_segment=10,
    end_segment=14,
    text="\n".join(f"[{i}] 00:00:{i:02d}–00:00:{i + 1:02d} line {i}" for i in range(10, 15)),
)

SOLAR = {"title": "Solar", "summary": "Cheap power.", "start_segment": 10, "end_segment": 12}
SLEEP = {"title": "Sleep", "summary": "Rest well.", "start_segment": 13, "end_segment": 14}


def _answer(*topics: dict) -> str:
    return json.dumps({"topics": list(topics)})


def _extractor(responses: list[str], max_retries: int = 2):
    llm = FakeLLMClient(responses)
    return LLMTopicExtractor(llm, LLMConfig(max_retries=max_retries)), llm


# --- parsing -------------------------------------------------------------------------


def test_parse_valid_answer():
    assert parse_topics(_answer(SOLAR, SLEEP), WINDOW) == [Topic(**SOLAR), Topic(**SLEEP)]


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps([SOLAR]),
        f"```json\n{_answer(SOLAR)}\n```",
        f"  ```\n{_answer(SOLAR)}\n```  ",
    ],
)
def test_parse_accepts_bare_list_and_code_fences(raw: str):
    assert parse_topics(raw, WINDOW) == [Topic(**SOLAR)]


def test_parse_empty_list():
    assert parse_topics('{"topics": []}', WINDOW) == []


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Sure! Here are the topics", "not valid JSON"),
        ('{"result": 1}', "form"),
        (_answer({**SOLAR, "start_segment": 9}), "start_segment 9 is not in this transcript"),
        (_answer({**SOLAR, "end_segment": 15}), "use ids from 10 to 14"),
        (_answer({**SOLAR, "start_segment": 12, "end_segment": 11}), "end_segment < start"),
        (_answer({"title": "x", "summary": "y", "start_segment": 10}), "end_segment"),
        (_answer({**SOLAR, "title": ""}), "title"),
    ],
)
def test_parse_rejects_invalid_answers(raw: str, message: str):
    with pytest.raises(InvalidTopicsError, match=message):
        parse_topics(raw, WINDOW)


# --- extraction with retries -----------------------------------------------------------


def test_extract_valid_response():
    extractor, llm = _extractor([_answer(SOLAR, SLEEP)])
    assert extractor.extract(WINDOW) == [Topic(**SOLAR), Topic(**SLEEP)]

    (call,) = llm.calls
    assert call.system == prompts.SYSTEM_PROMPT
    assert WINDOW.text in call.prompt
    assert "segments 10 to 14" in call.prompt
    assert call.json_schema == prompts.TOPICS_SCHEMA


def test_malformed_json_is_retried_with_the_parse_error():
    extractor, llm = _extractor(["{not json", _answer(SOLAR)])
    assert extractor.extract(WINDOW) == [Topic(**SOLAR)]

    retry = llm.calls[1].prompt
    assert retry.startswith(llm.calls[0].prompt)
    assert "not valid JSON" in retry
    assert "{not json" in retry


def test_out_of_range_segment_id_is_retried():
    extractor, llm = _extractor([_answer({**SOLAR, "end_segment": 99}), _answer(SOLAR)])
    assert extractor.extract(WINDOW) == [Topic(**SOLAR)]
    assert "end_segment 99 is not in this transcript" in llm.calls[1].prompt


def test_gives_up_after_max_retries():
    extractor, llm = _extractor(["nope", "still nope", "nope again", _answer(SOLAR)])
    with pytest.raises(TopicExtractionError, match="10–14.*after 3 attempts.*not valid JSON"):
        extractor.extract(WINDOW)
    assert len(llm.calls) == 3  # 1 attempt + 2 retries


def test_no_retries_configured():
    extractor, llm = _extractor(["nope", _answer(SOLAR)], max_retries=0)
    with pytest.raises(TopicExtractionError, match="after 1 attempts"):
        extractor.extract(WINDOW)
    assert len(llm.calls) == 1


def test_llm_errors_are_not_retried():
    extractor, llm = _extractor([])
    with pytest.raises(LLMError):
        extractor.extract(WINDOW)
    assert len(llm.calls) == 1


def test_golden_topics_round_trip(golden_job: Job):
    """The golden topics, as an LLM answer for a window over the whole golden transcript."""
    transcript = golden_job.read_transcript()
    golden = golden_job.read_topics()
    window = TranscriptWindow(
        start_segment=0, end_segment=len(transcript.segments) - 1, text="(transcript)"
    )
    answer = _answer(*(t.model_dump() for t in golden))
    extractor, _ = _extractor([answer])
    assert extractor.extract(window) == golden
    assert extractor.prompt_version == prompts.PROMPT_VERSION

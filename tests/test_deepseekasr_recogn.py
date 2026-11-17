"""DeepSeek ASR channel: registration, request shape and error handling. HTTP is mocked."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from videotrans import get_class
from videotrans.configure.config import params
from videotrans.configure.excepts import StopTask
from videotrans.recognition import (
    DEEPSEEK_ASR,
    ID_NAME_DICT,
    _deepseekasr,
    get_model_by_type,
    is_allow_lang,
)
from videotrans.recognition._deepseekasr import DEEPSEEK_ASR_URL, DeepSeekASRRecogn


def _clips(tmp_path, n=2):
    raws = []
    for i in range(n):
        f = tmp_path / f"audio_{i}.wav"
        f.write_bytes(b"RIFF" + bytes(2048))
        raws.append({"line": i + 1, "text": "", "start_time": i * 1000, "end_time": i * 1000 + 900,
                     "startraw": "", "endraw": "", "time": "", "filename": str(f)})
    return raws


def _resp(status=200, body=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or {}
    r.text = text
    return r


@pytest.fixture
def key():
    old = params.get("deepseek_key", "")
    params["deepseek_key"] = "sk_test_x"
    yield
    params["deepseek_key"] = old


def _run(tmp_path, side_effect, detect_language="zh-cn", n=2):
    rec = DeepSeekASRRecogn(detect_language=detect_language, recogn_type=DEEPSEEK_ASR)
    with patch.object(DeepSeekASRRecogn, "cut_audio", return_value=_clips(tmp_path, n)), \
            patch.object(DeepSeekASRRecogn, "_exit", return_value=False), \
            patch.object(_deepseekasr.requests, "post", side_effect=side_effect):
        return rec._exec()


def test_registered():
    assert ID_NAME_DICT[DEEPSEEK_ASR].key_name == "deepseek_key"
    assert ID_NAME_DICT[DEEPSEEK_ASR].win == "deepseekasr"
    assert get_class(DEEPSEEK_ASR, "recognition", ID_NAME_DICT) is DeepSeekASRRecogn
    assert get_model_by_type(DEEPSEEK_ASR) == ["deepseek-asr"]
    assert is_allow_lang("zh", DEEPSEEK_ASR) is True


def test_request_shape_and_srt_mapping(tmp_path, key):
    post = MagicMock(side_effect=[
        _resp(body={"text": " 你好。 ", "duration_ms": 900}),
        _resp(body={"text": "再见。", "duration_ms": 900}),
    ])
    res = _run(tmp_path, post)
    assert post.call_count == 2
    call = post.call_args_list[0]
    assert call.args[0] == DEEPSEEK_ASR_URL
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk_test_x"
    assert call.kwargs["files"]["file"][0] == "audio_0.wav"
    assert call.kwargs["files"]["file"][2] == "audio/wav"
    assert [it["text"] for it in res] == ["你好。", "再见。"]
    assert res[1]["start_time"] == 1000 and res[1]["end_time"] == 1900


def test_empty_text_is_dropped(tmp_path, key):
    post = MagicMock(side_effect=[
        _resp(body={"text": "  "}),
        _resp(body={"text": "好的"}),
    ])
    res = _run(tmp_path, post)
    assert [it["text"] for it in res] == ["好的"]


def test_missing_key_stops(tmp_path):
    params["deepseek_key"] = ""
    with pytest.raises(StopTask, match="Api Key"):
        _run(tmp_path, MagicMock())


def test_no_speech_clip_is_skipped(tmp_path, key):
    post = MagicMock(side_effect=[
        _resp(400, text="bad audio"),
        _resp(body={"text": "继续"}),
    ])
    res = _run(tmp_path, post)
    assert [it["text"] for it in res] == ["继续"]


def test_invalid_key_stops(tmp_path, key):
    post = MagicMock(return_value=_resp(401, text="unauthorized"))
    with pytest.raises(StopTask, match="platform.deepseek.com"):
        _run(tmp_path, post)


def test_too_large_clip_stops(tmp_path, key):
    post = MagicMock(return_value=_resp(413, text="too large"))
    with pytest.raises(StopTask, match="8 MiB"):
        _run(tmp_path, post)


def test_connection_error_stops(tmp_path, key):
    post = MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(StopTask, match="api.deepseek.com"):
        _run(tmp_path, post)

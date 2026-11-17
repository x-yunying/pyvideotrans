# DeepSeek Speech to Text (https://api-docs.deepseek.com/guides/asr)
#
# DeepSeek ASR returns the transcript of the whole uploaded clip and has no
# word/segment timestamps, so we first split the audio with the shared VAD
# helper (`cut_audio`), then request each short clip separately. The VAD clip
# boundaries provide the subtitle timing.
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Union

import requests
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from videotrans.configure.config import logger, params, settings, tr
from videotrans.configure.excepts import NO_RETRY_EXCEPT, SpeechToTextError, StopTask
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem

DEEPSEEK_ASR_URL = "https://api.deepseek.com/v1/tools/asr"
DEEPSEEK_ASR_KEY_URL = "https://platform.deepseek.com/api_keys"
# The API accepts at most 8 MiB and 120 s of audio per request.
DEEPSEEK_ASR_MAX_BYTES = 8 * 1024 * 1024


@dataclass(repr=False)
class DeepSeekASRRecogn(BaseRecogn):
    """DeepSeek ASR channel.

    Reuses ``deepseek_key`` from the translation settings, so a single
    DeepSeek API key is shared by both channels.
    """

    def _transcribe(self, audio_path: str) -> str:
        api_key = params.get("deepseek_key", "").strip()
        if not api_key:
            raise StopTask(tr("api key must be filled in"))

        if os.path.getsize(audio_path) > DEEPSEEK_ASR_MAX_BYTES:
            raise StopTask(
                f"DeepSeek ASR: audio clip {os.path.basename(audio_path)} is larger than 8 MiB. "
                f"Please lower `max_speech_duration_s` in the advanced settings."
            )

        try:
            with open(audio_path, "rb") as f:
                res = requests.post(
                    DEEPSEEK_ASR_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                    timeout=300,
                )
        except requests.exceptions.ConnectionError as e:
            raise StopTask(f'{tr("Unable to connect to API", DEEPSEEK_ASR_URL)}\n{e}') from e

        if res.status_code == 401:
            raise StopTask(
                f"DeepSeek ASR: the API key is invalid. Get a key at {DEEPSEEK_ASR_KEY_URL}"
            )
        if res.status_code == 413:
            raise StopTask(
                "DeepSeek ASR: the audio clip is larger than 8 MiB or longer than 120 s. "
                "Please lower `max_speech_duration_s` in the advanced settings."
            )
        if res.status_code == 400:
            # A VAD clip that is only noise / silence. Skip it and keep the rest.
            logger.warning(f"DeepSeek ASR skipped {os.path.basename(audio_path)}: {res.text[:300]}")
            return ""
        # 408/429/5xx are transient; raise so tenacity retries this clip.
        res.raise_for_status()
        return (res.json().get("text") or "").strip()

    @retry(
        retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),
        stop=stop_after_attempt(int(settings.get("retry_nums", 2))),
        wait=wait_fixed(2),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def _transcribe_with_retry(self, audio_path: str) -> str:
        return self._transcribe(audio_path)

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return

        raws = self.cut_audio()
        if not raws:
            raise SpeechToTextError(
                tr("No speech was detected, please make sure there is human speech in the selected audio/video")
            )

        for it in raws:
            if self._exit():
                return
            text = self._transcribe_with_retry(it["filename"])
            if not text:
                continue
            it["text"] = text
            self.signal(text=text + "\n", type="subtitle")
            if self.asr_wait and self.asr_wait > 0:
                time.sleep(self.asr_wait)

        return [it for it in raws if it["text"]]

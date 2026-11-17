import base64
import json
import logging,time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import requests

from videotrans.configure.excepts import SpeechToTextError, StopTask
from videotrans.configure.config import params, logger
from videotrans.configure import config
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_srt import ms_to_time_string

@dataclass(repr=False)
class Qwen3ASRRecogn(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()
        spaceid=params.get('bailian_spaceid', '').strip()
        # 未填写业务空间ID时使用百炼默认地址，否则 api_url 为空字符串，请求地址无协议头导致报错
        self.api_url = 'https://dashscope.aliyuncs.com/api/v1'
        if spaceid and  not spaceid.startswith('http'):
            self.api_url = f'https://{spaceid}.cn-beijing.maas.aliyuncs.com/api/v1'
        elif spaceid and spaceid.startswith('http'):
            self.api_url = spaceid.strip().strip('/')
        # qwen-audio-3.1-asr-flash 支持一次提交完整音频，并返回句级/词级时间戳和说话人ID。
        # 5分钟以内直接整段识别，可获得比 VAD 切段更准确的断句和说话人归属。
        self._asr_speaker_enabled = self.max_speakers > -1
        # 单人模式仍使用 VAD 分段；启用“识别说话人”（多人）时统一使用词级/句级时间戳。
        # 超过 5 分钟会分块请求 3.1，再由本地说话人分离统一分配 spk，保证同一人跨块一致。
        self._whole_audio = self.model_name == 'qwen-audio-3.1-asr-flash' and self._asr_speaker_enabled
        self._chunked_audio = self._whole_audio and self._audio_duration() > 295
        # 分块模式下 API 返回的 speaker_id 只在块内有效，因此不直接使用，交给本地说话人分离
        self._use_api_speaker = self._asr_speaker_enabled and not self._chunked_audio
        logger.debug(
            f'Qwen-ASR 分段模式: '
            f'{"词级时间戳(多人)-分块" if self._chunked_audio else ("词级时间戳(多人)" if self._whole_audio else "VAD")}'
        )
        self.raws = [] if self._whole_audio else self.cut_audio()

    def _audio_duration(self) -> float:
        try:
            import wave
            with wave.open(self.audio_file, 'rb') as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    def _make_item(self, text: str, start: int, end: int, spk: str) -> SrtItem:
        start = max(0, int(start))
        end = max(start, int(end))
        startraw = ms_to_time_string(ms=start)
        endraw = ms_to_time_string(ms=end)
        return SrtItem(
            line=0,
            text=text,
            start_time=start,
            end_time=end,
            startraw=startraw,
            endraw=endraw,
            time=f'{startraw} --> {endraw}',
            spk=spk,
            filename=''
        )

    def _sentence_to_items(self, sentence: dict, offset: int = 0) -> List[SrtItem]:
        text = (sentence.get('text') or '').strip()
        if not text:
            return []
        speaker_id = sentence.get('speaker_id')
        spk = f'spk{speaker_id}' if self._use_api_speaker and speaker_id is not None else ''
        words = sentence.get('words') or []
        if not words:
            start = (sentence.get('begin_time') or 0) + offset
            end = (sentence.get('end_time') or 0) + offset or start
            return [self._make_item(text, start, end, spk)]

        # 按词级时间戳重新分行，避免一整句过长
        chunks = []
        cur = []
        cur_len = 0
        for w in words:
            wtext = (w.get('text') or '') + (w.get('punctuation') or '')
            add_len = len(wtext.strip())
            if cur and cur_len + add_len > self.maxlen:
                chunks.append(cur)
                cur = []
                cur_len = 0
            cur.append(w)
            cur_len += add_len
            punct = (w.get('punctuation') or '').strip()
            if punct and cur_len >= max(12, self.maxlen // 2):
                chunks.append(cur)
                cur = []
                cur_len = 0
        if cur:
            chunks.append(cur)

        items = []
        for chunk in chunks:
            _text = ''.join((w.get('text') or '') + (w.get('punctuation') or '') for w in chunk).strip()
            if not _text:
                continue
            start = (chunk[0].get('begin_time') or 0) + offset
            end = (chunk[-1].get('end_time') or 0) + offset or start
            items.append(self._make_item(_text, start, end, spk))
        return items

    def _request_qwen31(self, api_key: str, data_uri: str) -> List[dict]:
        url = self.api_url.rstrip('/') + '/services/aigc/multimodal-generation/generation'
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": data_uri},
                    }]
                }]
            },
            "parameters": {
                "format": "wav",
                "sample_rate": "16000",
                # 开启后返回 output.sentences；分块模式下 speaker_id 仅块内有效，另行本地分离
                "speaker_diarization_enabled": True,
            },
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "disable",
            },
            json=payload,
            timeout=(30, 600),
            proxies={"https": "", "http": ""}
        )
        if response.status_code in (401, 403, 404):
            raise StopTask(response.text)
        if response.status_code != 200:
            raise SpeechToTextError(f'Qwen-Audio-3.1-ASR: {response.status_code} {response.text}')

        data = response.json()
        output = data.get('output', {})
        inner = output.get('output', output)
        sentences = inner.get('sentences') or []
        if not sentences and inner.get('sentence'):
            sentences = [inner['sentence']]
        return sentences

    def _qwen31_whole_audio(self, api_key: str) -> List[SrtItem]:
        if not self._chunked_audio:
            data_uri = 'data:audio/wav;base64,' + base64.b64encode(Path(self.audio_file).read_bytes()).decode()
            sentences = self._request_qwen31(api_key, data_uri)
            items = []
            for sentence in sentences:
                items.extend(self._sentence_to_items(sentence))
            if not items:
                raise SpeechToTextError('Qwen-Audio-3.1-ASR 未返回有效句子')
            return items
        return self._qwen31_chunked_audio(api_key)

    def _qwen31_chunked_audio(self, api_key: str) -> List[SrtItem]:
        import shutil
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(self.audio_file)
        total_ms = len(audio)
        # 单次请求上限 5 分钟，留出余量；块间保留 8 秒重叠用于边界句子补全
        chunk_ms = 270000
        overlap_ms = 8000
        tmp_dir = Path(f'{config.TEMP_DIR}/qwen31chunks_{self.uuid or "none"}')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        start = 0
        idx = 0
        last_end = -1
        items: List[SrtItem] = []
        try:
            while start < total_ms:
                end = min(start + chunk_ms, total_ms)
                chunk_file = tmp_dir / f'chunk_{idx}.wav'
                audio[start:end].export(chunk_file, format='wav')
                data_uri = 'data:audio/wav;base64,' + base64.b64encode(chunk_file.read_bytes()).decode()
                sentences = self._request_qwen31(api_key, data_uri)
                for sentence in sentences:
                    for it in self._sentence_to_items(sentence, offset=start):
                        # 丢弃与上一块重叠导致的重复句
                        if last_end >= 0 and it['start_time'] < last_end - 300:
                            continue
                        items.append(it)
                        last_end = max(last_end, it['end_time'])
                if end >= total_ms:
                    break
                start = end - overlap_ms
                idx += 1
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not items:
            raise SpeechToTextError('Qwen-Audio-3.1-ASR 分块识别未返回有效句子')
        logger.debug(f'Qwen-Audio-3.1 分块识别完成，共 {len(items)} 句')
        return items

    def _post_fix(self, res: List[SrtItem]) -> List[SrtItem]:
        out = super()._post_fix(res)
        # 只有直接采用 API speaker_id 时才写入 speaker.json；
        # 分块模式的 speaker_id 不跨块一致，交给本地说话人分离。
        if self._use_api_speaker and self.cache_folder and out:
            try:
                Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
                Path(f'{self.cache_folder}/speaker.json').write_text(
                    json.dumps([it.get('spk') or 'spk0' for it in out], ensure_ascii=False),
                    encoding='utf-8'
                )
            except Exception as e:
                logger.warning(f'写入 qwen-audio-3.1 说话人结果失败，忽略: {e}')
        return out

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit(): return
        # 发送请求
        api_key=params.get('bailian_key','')
        model=self.model_name
        if model == 'qwen-audio-3.1-asr-flash' and self._whole_audio:
            try:
                return self._qwen31_whole_audio(api_key)
            except StopTask:
                raise
            except Exception as e:
                logger.warning(f'Qwen-Audio-3.1 整段识别失败，回退到 VAD 分段识别: {e}')
                self._whole_audio = False
                self.raws = self.cut_audio()
                return self._audio_funasr_flash(api_key, model)
        # Qwen-Audio ASR 的 3.0 / 3.1 和 Fun-ASR Flash 使用相同的 HTTP 接口
        if model.startswith(('qwen-audio-3.0-asr-flash', 'qwen-audio-3.1-asr-flash', 'fun-asr-flash')):
            return self._audio_funasr_flash(api_key,model)

        error=""
        ok_nums=0
        # Qwen3-ASR 走 DashScope SDK；Qwen-Audio 3.x 走上面的 HTTP 接口
        import dashscope
        from dashscope.common.error import AuthenticationError
        dashscope.base_http_api_url = self.api_url
        for i, it in enumerate(self.raws):
            try:
                response = dashscope.MultiModalConversation.call(
                    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
                    api_key=api_key,
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"audio": it['filename']},
                        ]
                    }],
                    result_format="message",
                    asr_options={
                        "language": None if self.detect_language=='auto' else  self.detect_language.split('-')[0], # 可选，若已知音频的语种，可通过该参数指定待识别语种，以提升识别准确率
                        "enable_lid": True,
                        "enable_itn": True
                    }
                )
            except AuthenticationError as e:
                raise StopTask(str(e))

            if not hasattr(response, 'output') or not hasattr(response.output, 'choices') or not response.output.choices:
                error=f'{response.code}:{response.message}'
                continue

            ok_nums+=1
            txt=''
            for t in response.output.choices[0]['message']['content']:
                txt += t['text']
            self.raws[i]['text'] = txt
            self.signal(text=f"{txt}\n",type="subtitle")
            if self.asr_wait>0:
                time.sleep(self.asr_wait)

        if ok_nums==0:
            raise SpeechToTextError(error)
        return self.raws


    # 针对 qwen-audio-3.0-asr-flash 和 fun-asr-flash-2026-06-15
    def _audio_funasr_flash(self,api_key,model):
        if self._exit(): return
        # 发送请求

        error=""
        ok_nums=0
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }
        # /services/aigc/multimodal-generation/generation
        if self.api_url.endswith('/v1'):
            self.api_url+='/services/aigc/multimodal-generation/generation'

        for i, it in enumerate(self.raws):
            base64_str = base64.b64encode(Path(it['filename']).read_bytes()).decode()
            data_uri = f"data:audio/wav;base64,{base64_str}"

            payload = {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": data_uri,
                                    },
                                }
                            ],
                        }
                    ]
                },
                "parameters": {
                    "format": "wav",
                    "sample_rate": "16000",
                },
            }


            try:
                # 带 API Key 访问公网，保持证书校验
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=(30, 300), proxies={"https":"","http":""})
            except requests.exceptions.RequestException as e:
                error=str(e)
                continue
            if response.status_code in [401,403,404]:
                raise StopTask(response.text)
            if response.status_code in [400,422]:
                # 单个片段被内容审核拦截时跳过该片段，其他参数错误仍中止任务
                if 'DataInspectionFailed' in response.text:
                    error=response.text
                    continue
                raise StopTask(response.text)
            if response.status_code!=200:
                error=response.text
                continue
            try:
                data=response.json()
            except json.JSONDecodeError:
                error=response.text
                continue
            if not data.get('output') or not data.get('output',{}).get('text'):
                error=response.text
                continue
            ok_nums+=1
            _text=data.get('output').get('text')
            self.signal(text=_text+"\n",type="subtitle")
            self.raws[i]['text'] = _text
            if self.asr_wait>0:
                time.sleep(self.asr_wait)

        if ok_nums==0:
            raise SpeechToTextError(error)
        return self.raws
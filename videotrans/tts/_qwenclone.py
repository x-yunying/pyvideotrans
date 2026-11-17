import base64
import hashlib
import logging
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log

from videotrans.configure import config
from videotrans.configure.config import params, logger, settings
from videotrans.configure.excepts import NO_RETRY_EXCEPT, StopTask
from videotrans.tts._base import BaseTTS
from videotrans.util.help_ffmpeg import runffmpeg
from videotrans.util.help_misc import vail_file, get_md5


def _file_md5(path: str) -> str:
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5.update(chunk)
    return md5.hexdigest()


# 阿里云百炼千问声音复刻：先创建音色，再用 qwen3-tts-vc 合成
# https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning
@dataclass(repr=False)
class QwenCloneTTS(BaseTTS):

    def __post_init__(self):
        super().__post_init__()
        self.api_key = str(params.get('qwenclone_key', '')).strip()
        self.vc_model = str(params.get('qwenclone_model', 'qwen-audio-3.1-tts-plus')).strip()
        self.base_url = self._base_url()

        self._lock = threading.Lock()
        self._voices: Dict[str, str] = {}
        # 每个说话人保留少量参考音频候选，按长度优先尝试
        self._ref_by_key: Dict[str, list] = self._pick_reference_audio()

    def _base_url(self) -> str:
        spaceid = str(params.get('qwenclone_spaceid', '')).strip().rstrip('/')
        if not spaceid:
            return 'https://dashscope.aliyuncs.com/api/v1'
        if spaceid.startswith('http'):
            return spaceid if spaceid.endswith('/api/v1') else spaceid + '/api/v1'
        if '.aliyuncs.com' in spaceid:
            base = 'https://' + spaceid.rstrip('/')
            return base if base.endswith('/api/v1') else base + '/api/v1'
        return f'https://{spaceid}.cn-beijing.maas.aliyuncs.com/api/v1'

    @staticmethod
    def _duration(path: str) -> float:
        try:
            with wave.open(path, 'rb') as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    def _pick_reference_audio(self) -> Dict[str, list]:
        result: Dict[str, list] = {}
        for item in self.queue_tts:
            if str(item.get('role', '')).strip().lower() != 'clone':
                continue
            ref_wav = item.get('ref_wav')
            if not ref_wav or not Path(ref_wav).exists():
                continue
            # speaker.json 对齐到字幕行，由 _stage_dubbing 写入 spk。
            # 没有 spk 时，整个视频使用同一个音色。
            key = str(item.get('spk') or self.uuid or 'default')
            result.setdefault(key, []).append(
                (self._duration(ref_wav), ref_wav, str(item.get('ref_language') or ''))
            )
        # 优先尝试较长的参考音频；单条音频可能因纯音乐/噪声导致云端 ASR 失败，
        # 因此保留多条候选，失败时自动换下一条。
        for refs in result.values():
            refs.sort(key=lambda x: x[0], reverse=True)
            del refs[8:]
        return result

    def _build_combined_reference(self, refs: list):
        """把同一说话人的多段短句拼接成一段足够长的参考音频，避免 AudioShortError。"""
        try:
            from pydub import AudioSegment
            combined = AudioSegment.empty()
            used = []
            for _, ref_wav, ref_language in refs:
                try:
                    combined += AudioSegment.from_file(ref_wav)
                    used.append((ref_wav, ref_language))
                except Exception:
                    continue
                if len(combined) >= 15000:
                    break
            if len(combined) < 3000 or not used:
                return None, []
            combined = combined.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            if len(combined) > 30000:
                combined = combined[:30000]
            out = Path(f'{config.TEMP_DIR}/{self.uuid}/qwenclone-combined-{get_md5("|".join(x[0] for x in used))}.wav')
            out.parent.mkdir(parents=True, exist_ok=True)
            combined.export(out, format='wav')
            return out.as_posix(), used
        except Exception as e:
            logger.warning(f'Qwen-TTS 拼接参考音频失败，将尝试单段音频: {e}')
            return None, []

    def _prepare_reference(self, ref_wav: str) -> str:
        # 千问要求单声道、采样率 >= 24kHz；重新编码后同时满足格式和大小要求
        out = f'{config.TEMP_DIR}/{self.uuid}/qwenclone-ref-{get_md5(ref_wav)}.wav'
        if not Path(out).exists() or Path(out).stat().st_size == 0:
            try:
                runffmpeg([
                    '-y', '-i', ref_wav,
                    '-t', '30',
                    '-ac', '1',
                    '-ar', '24000',
                    '-c:a', 'pcm_s16le',
                    out
                ])
            except Exception as e:
                raise StopTask(f'Qwen-TTS 参考音频格式转换失败: {e}') from e
        if not Path(out).exists() or Path(out).stat().st_size == 0:
            raise StopTask(f'Qwen-TTS 参考音频不存在: {out}')
        return out

    def _is_qwen3_vc(self) -> bool:
        return 'qwen3-tts-vc' in self.vc_model

    def _scope_tag(self) -> str:
        # 不同账号/地域/目标模型创建的音色不能混用
        account = hashlib.md5(f'{self.api_key}|{self.base_url}'.encode('utf-8')).hexdigest()[:8]
        model_tag = hashlib.md5(self.vc_model.encode('utf-8')).hexdigest()[:6]
        return f'{account}_{model_tag}'

    def _create_voice(self, ref_wav: str, ref_language: str = '') -> str:
        fixed = self._prepare_reference(ref_wav)
        digest = _file_md5(fixed)
        cache_key = f'qwenclone_voice_{digest}_{self._scope_tag()}'
        cached = params.get(cache_key)
        if cached:
            logger.debug(f'Qwen-TTS 复用已创建的音色: {cached}')
            return cached

        audio = Path(fixed).read_bytes()
        data_uri = 'data:audio/wav;base64,' + base64.b64encode(audio).decode()
        # 官方限制是 Base64 编码后小于 10MB
        if len(data_uri) > 10 * 1024 * 1024:
            raise StopTask('Qwen-TTS 参考音频编码后超过 10MB')

        if self._is_qwen3_vc():
            payload = {
                "model": "qwen-voice-enrollment",
                "input": {
                    "action": "create",
                    # 创建音色和合成语音必须使用同一个模型
                    "target_model": self.vc_model,
                    # 仅允许数字、大小写字母和下划线，不超过 16 个字符
                    "preferred_name": 'qwen_' + digest[:11],
                    "audio": {"data": data_uri}
                }
            }
        else:
            # Qwen-Audio-TTS / CosyVoice 复刻音色接口
            input_obj = {
                "action": "create_voice",
                "target_model": self.vc_model,
                # prefix 仅允许数字和字母，不超过 10 个字符
                "prefix": 'q' + digest[:8],
                "audio": data_uri
            }
            # 参考音频自身的语种，而不是目标配音语种
            lang = (ref_language or self.language or '').split('-')[0]
            if lang in ('zh', 'en'):
                input_obj['language_hints'] = [lang]
            payload = {
                "model": "voice-enrollment",
                "input": input_obj
            }
        url = self.base_url.rstrip('/') + '/services/audio/tts/customization'
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120
        )
        if response.status_code != 200:
            raise StopTask(f'Qwen-TTS 创建音色失败: {response.status_code} {response.text}')

        try:
            data = response.json()
            output = data.get('output', {})
            voice = output.get('voice') or output.get('voice_id')
            if not voice:
                voice_list = output.get('voice_list') or []
                voice = voice_list[0].get('voice') if voice_list else None
        except Exception:
            voice = None
        if not voice:
            raise StopTask(f'Qwen-TTS 创建音色返回异常: {response.text}')

        params[cache_key] = voice
        # 记录参考音频 -> 音色，后续优先复用已成功的片段，避免再次触发失败的 ASR
        params[f'qwenclone_refvoice_{get_md5(ref_wav)}_{self._scope_tag()}'] = voice
        params.save()
        logger.debug(f'Qwen-TTS 创建音色成功: {voice}')
        return voice

    def _get_voice(self, key: str) -> str:
        with self._lock:
            if key in self._voices:
                return self._voices[key]
            refs = self._ref_by_key.get(key)
            if not refs:
                raise StopTask('Qwen-TTS clone 模式缺少参考音频，请检查字幕时长或说话人识别结果')

            # 优先使用曾经成功创建过音色的参考片段
            for _, ref_wav, _ in refs:
                cached = params.get(f'qwenclone_refvoice_{get_md5(ref_wav)}_{self._scope_tag()}')
                if cached:
                    self._voices[key] = cached
                    return cached

            last_error = None
            # 词级断句后单句可能过短，先拼接同一说话人的多段语音，保证音频长度
            combined_wav, used_refs = self._build_combined_reference(refs)
            if combined_wav:
                try:
                    voice = self._create_voice(combined_wav, used_refs[0][1] if used_refs else '')
                    self._voices[key] = voice
                    for ref_wav, _ in used_refs:
                        params[f'qwenclone_refvoice_{get_md5(ref_wav)}_{self._scope_tag()}'] = voice
                    params.save()
                    return voice
                except StopTask as e:
                    last_error = e
                    logger.warning(f'Qwen-TTS 拼接参考音频创建音色失败，尝试单段音频: {e}')

            for _, ref_wav, ref_language in refs:
                try:
                    voice = self._create_voice(ref_wav, ref_language)
                    self._voices[key] = voice
                    return voice
                except StopTask as e:
                    last_error = e
                    logger.warning(f'Qwen-TTS 参考音频创建音色失败，尝试下一条: {ref_wav} {e}')

            raise StopTask(
                'Qwen-TTS 创建音色失败：参考音频未通过云端语音识别，可能是纯音乐、歌声或噪声。'
                '请开启“人声背景声分离”后重试，或更换包含清晰人声的视频。'
                f' 最后一次错误: {last_error}'
            )

    @retry(retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),
           stop=stop_after_attempt(settings.get('retry_nums') or 2),
           wait=wait_fixed(2),
           before=before_log(logger, logging.INFO),
           after=after_log(logger, logging.INFO))
    def _run(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, None]:
        if vail_file(data_item['filename']):
            return
        if not self.api_key:
            raise StopTask('请先填写阿里云百炼 API Key')

        role = str(data_item.get('role', '')).strip()
        if role.lower() == 'clone':
            key = str(data_item.get('spk') or self.uuid or 'default')
            voice = self._get_voice(key)
            model = self.vc_model
        else:
            if not role or role == 'No':
                return '未选择配音角色'
            # qwenclone_roles 中填写的是已创建的音色 ID，同样必须用 VC 模型
            voice = role
            model = self.vc_model

        # 直接调用 HTTP 接口，避免额外依赖 dashscope SDK
        text = data_item.get('text', '').strip()
        if self._is_qwen3_vc():
            url = self.base_url.rstrip('/') + '/services/aigc/multimodal-generation/generation'
            input_obj = {
                "text": text,
                "voice": voice,
                "language_type": "Auto"
            }
        else:
            url = self.base_url.rstrip('/') + '/services/audio/tts/SpeechSynthesizer'
            input_obj = {
                "text": text,
                "voice": voice,
                "format": "wav",
                "sample_rate": 24000
            }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "input": input_obj
            },
            timeout=120
        )
        if response.status_code != 200:
            raise StopTask(f'Qwen-TTS 合成失败: {response.status_code} {response.text}')
        try:
            data = response.json()
            audio = data.get('output', {}).get('audio')
            if isinstance(audio, list):
                audio = audio[0] if audio else None
            audio_url = audio.get('url') if isinstance(audio, dict) else None
        except Exception:
            audio_url = None
        if not audio_url:
            return f'Qwen-TTS 未返回音频: {response.text}'

        res = requests.get(audio_url, timeout=120)
        res.raise_for_status()
        tmp_wav = data_item['filename'] + '.wav'
        Path(tmp_wav).write_bytes(res.content)
        self.convert_to_wav(tmp_wav, data_item['filename'])

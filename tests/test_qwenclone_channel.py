import shutil
import wave
from pathlib import Path

from videotrans import tts
from videotrans.configure.config import params
from videotrans.tts import _qwenclone


def _wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    with wave.open(path.as_posix(), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00\x00' * int(sample_rate * seconds))


def test_channel_is_registered():
    assert tts.QWEN_CLONE_TTS in tts.SUPPORT_CLONE
    provider = tts.ID_NAME_DICT[tts.QWEN_CLONE_TTS]
    assert provider.imp == '._qwenclone'
    assert provider.key_name == 'qwenclone_key'
    assert provider.win == 'qwenclone'


def test_clone_role_is_available():
    from videotrans.util.help_role import role_menu

    assert 'clone' in role_menu(tts.QWEN_CLONE_TTS)


def test_clone_creates_one_voice_per_speaker(monkeypatch, tmp_path):
    monkeypatch.setattr(params, 'qwenclone_key', 'sk-test', raising=False)
    monkeypatch.setattr(params, 'qwenclone_spaceid', '', raising=False)
    monkeypatch.setattr(params, 'qwenclone_model', 'qwen3-tts-vc-2026-01-22', raising=False)
    monkeypatch.setattr(params, 'save', lambda *args, **kwargs: None)

    short_ref = tmp_path / 'short.wav'
    long_ref = tmp_path / 'long.wav'
    _wav(short_ref, 1)
    _wav(long_ref, 3)

    queue = [
        {
            'line': 1,
            'text': 'line 1',
            'role': 'clone',
            'spk': 'spk0',
            'ref_wav': short_ref.as_posix(),
            'filename': (tmp_path / 'line1.wav').as_posix(),
        },
        {
            'line': 2,
            'text': 'line 2',
            'role': 'clone',
            'spk': 'spk0',
            'ref_wav': long_ref.as_posix(),
            'filename': (tmp_path / 'line2.wav').as_posix(),
        },
    ]

    ffmpeg_calls = []

    def fake_runffmpeg(args, **kwargs):
        ffmpeg_calls.append(args)
        src = args[args.index('-i') + 1]
        dst = args[-1]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    monkeypatch.setattr(_qwenclone, 'runffmpeg', fake_runffmpeg)

    enrollment_posts = []
    synth_posts = []

    class JsonResponse:
        status_code = 200
        text = ''

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        if 'customization' in url:
            enrollment_posts.append((url, kwargs))
            return JsonResponse({'output': {'voice': 'qwen-voice-123'}})
        synth_posts.append((url, kwargs))
        return JsonResponse({'output': {'audio': {'url': 'https://example.com/result.wav'}}})

    monkeypatch.setattr(_qwenclone.requests, 'post', fake_post)

    class AudioResponse:
        content = b'fake audio'

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(_qwenclone.requests, 'get', lambda *args, **kwargs: AudioResponse())
    monkeypatch.setattr(
        _qwenclone.QwenCloneTTS,
        'convert_to_wav',
        staticmethod(lambda src, dst, extra=None: shutil.copy2(src, dst))
    )

    obj = _qwenclone.QwenCloneTTS(
        queue_tts=queue,
        language='zh',
        tts_type=tts.QWEN_CLONE_TTS,
        wait_sec=0,
    )
    obj._run(queue[0])
    obj._run(queue[1])

    # 同一个说话人只创建一次音色，并使用最长的那条参考音频
    assert len(enrollment_posts) == 1
    payload = enrollment_posts[0][1]['json']
    assert payload['model'] == 'qwen-voice-enrollment'
    assert payload['input']['target_model'] == 'qwen3-tts-vc-2026-01-22'
    assert len(payload['input']['preferred_name']) <= 16
    assert any('qwenclone-ref-' in args[-1] for args in ffmpeg_calls)
    assert any('-ar' in args and args[args.index('-ar') + 1] == '24000' for args in ffmpeg_calls)

    # 两行都使用同一个复刻音色，并保持创建/合成模型一致
    assert len(synth_posts) == 2
    for url, kwargs in synth_posts:
        assert url.endswith('/services/aigc/multimodal-generation/generation')
        assert kwargs['json']['model'] == 'qwen3-tts-vc-2026-01-22'
        assert kwargs['json']['input']['voice'] == 'qwen-voice-123'
    assert Path(queue[0]['filename']).exists()
    assert Path(queue[1]['filename']).exists()


def test_qwen_audio_tts_clone_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(params, 'qwenclone_key', 'sk-test', raising=False)
    monkeypatch.setattr(params, 'qwenclone_spaceid', '', raising=False)
    monkeypatch.setattr(params, 'qwenclone_model', 'qwen-audio-3.0-tts-flash', raising=False)
    monkeypatch.setattr(params, 'save', lambda *args, **kwargs: None)

    ref = tmp_path / 'ref.wav'
    _wav(ref, 5)

    def fake_runffmpeg(args, **kwargs):
        dst = args[-1]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args[args.index('-i') + 1], dst)

    monkeypatch.setattr(_qwenclone, 'runffmpeg', fake_runffmpeg)

    posts = []

    class JsonResponse:
        status_code = 200
        text = ''

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        if 'customization' in url:
            return JsonResponse({'output': {'voice_id': 'audio-voice-1'}})
        return JsonResponse({'output': {'audio': {'url': 'https://example.com/audio.wav'}}})

    monkeypatch.setattr(_qwenclone.requests, 'post', fake_post)

    class AudioResponse:
        content = b'fake audio'

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(_qwenclone.requests, 'get', lambda *args, **kwargs: AudioResponse())
    monkeypatch.setattr(
        _qwenclone.QwenCloneTTS,
        'convert_to_wav',
        staticmethod(lambda src, dst, extra=None: shutil.copy2(src, dst))
    )

    item = {
        'text': '测试',
        'role': 'clone',
        'spk': 'spk0',
        'ref_wav': ref.as_posix(),
        'filename': (tmp_path / 'out.wav').as_posix(),
    }
    obj = _qwenclone.QwenCloneTTS(
        queue_tts=[item],
        language='zh',
        tts_type=tts.QWEN_CLONE_TTS,
        wait_sec=0,
    )
    obj._run(item)

    enrollment = posts[0]
    assert enrollment[0].endswith('/services/audio/tts/customization')
    assert enrollment[1]['json']['model'] == 'voice-enrollment'
    assert enrollment[1]['json']['input']['action'] == 'create_voice'
    assert enrollment[1]['json']['input']['target_model'] == 'qwen-audio-3.0-tts-flash'
    assert len(enrollment[1]['json']['input']['prefix']) <= 10
    assert enrollment[1]['json']['input']['audio'].startswith('data:audio/wav;base64,')

    synthesis = posts[1]
    assert synthesis[0].endswith('/services/audio/tts/SpeechSynthesizer')
    assert synthesis[1]['json']['model'] == 'qwen-audio-3.0-tts-flash'
    assert synthesis[1]['json']['input']['voice'] == 'audio-voice-1'
    assert synthesis[1]['json']['input']['format'] == 'wav'
    assert Path(item['filename']).exists()


def test_voice_creation_falls_back_to_next_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(params, 'qwenclone_key', 'sk-test', raising=False)
    monkeypatch.setattr(params, 'qwenclone_spaceid', '', raising=False)
    monkeypatch.setattr(params, 'qwenclone_model', 'qwen-audio-3.0-tts-flash', raising=False)
    monkeypatch.setattr(params, 'save', lambda *args, **kwargs: None)

    long_ref = tmp_path / 'long.wav'
    short_ref = tmp_path / 'short.wav'
    _wav(long_ref, 3)
    _wav(short_ref, 1)

    def fake_runffmpeg(args, **kwargs):
        dst = args[-1]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args[args.index('-i') + 1], dst)

    monkeypatch.setattr(_qwenclone, 'runffmpeg', fake_runffmpeg)

    calls = []

    class JsonResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.text = str(payload)
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if 'customization' in url:
            if len(calls) == 1:
                # 最长参考音频可能因纯音乐/噪声导致 ASR 失败
                return JsonResponse(500, {
                    'code': 'InternalError',
                    'message': 'request asr failed, http code: 400'
                })
            return JsonResponse(200, {'output': {'voice_id': 'fallback-voice'}})
        return JsonResponse(200, {'output': {'audio': {'url': 'https://example.com/a.wav'}}})

    monkeypatch.setattr(_qwenclone.requests, 'post', fake_post)

    class AudioResponse:
        content = b'fake audio'

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(_qwenclone.requests, 'get', lambda *args, **kwargs: AudioResponse())
    monkeypatch.setattr(
        _qwenclone.QwenCloneTTS,
        'convert_to_wav',
        staticmethod(lambda src, dst, extra=None: shutil.copy2(src, dst))
    )

    item = {
        'text': '测试',
        'role': 'clone',
        'spk': 'spk0',
        'ref_wav': long_ref.as_posix(),
        'filename': (tmp_path / 'out.wav').as_posix(),
    }
    obj = _qwenclone.QwenCloneTTS(
        queue_tts=[item],
        language='zh',
        tts_type=tts.QWEN_CLONE_TTS,
        wait_sec=0,
    )
    # 手动把较短参考音频加入候选，模拟同一说话人有多条字幕片段
    obj._ref_by_key['spk0'] = [(3.0, long_ref.as_posix(), 'zh'), (1.0, short_ref.as_posix(), 'zh')]
    obj._run(item)

    assert len([c for c in calls if 'customization' in c[0]]) == 2
    synth = [c for c in calls if 'SpeechSynthesizer' in c[0]]
    assert len(synth) == 1
    assert synth[0][1]['json']['input']['voice'] == 'fallback-voice'

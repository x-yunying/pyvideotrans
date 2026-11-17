def openwin():
    import time
    from pathlib import Path

    from videotrans.winform import get_cls
    from videotrans.configure.config import tr, params, ROOT_DIR
    from videotrans.configure import config
    from videotrans.configure.constants import LISTEN_TEXT
    from videotrans.util.help_misc import show_error
    from videotrans.util.ListenVoice import ListenVoice
    from videotrans.winform._helpers import make_feed_tts

    winobj = get_cls(Path(__file__).stem)()
    winobj.update_ui()

    feed = make_feed_tts(winobj, "test")

    def _save_params():
        params["qwenclone_key"] = winobj.qwenclone_key.text().strip()
        params["qwenclone_spaceid"] = winobj.qwenclone_spaceid.text().strip()
        params["qwenclone_model"] = winobj.qwenclone_model.text().strip() or 'qwen-audio-3.0-tts-flash'
        params["qwenclone_roles"] = winobj.qwenclone_roles.text().strip()
        params.save()

    def test():
        _save_params()
        if not params.get('qwenclone_key'):
            return show_error(tr("Please input Secret"))

        ref_wav = f'{ROOT_DIR}/f5-tts/zh_female_nverguo.wav'
        if not Path(ref_wav).exists():
            for item in Path(f'{ROOT_DIR}/f5-tts').glob('*.wav'):
                ref_wav = item.as_posix()
                break
        if not Path(ref_wav).exists():
            return show_error(tr("No reference audio exists and cannot use clone function"))

        from videotrans import tts
        wk = ListenVoice(parent=winobj, queue_tts=[{
            "text": LISTEN_TEXT.get('zh', '你好啊，我亲爱的朋友，希望你的每一天都是美好愉快的！'),
            "role": "clone",
            "ref_wav": ref_wav,
            "ref_text": "",
            "filename": config.TEMP_DIR + f"/{time.time()}-qwenclone.wav",
            "tts_type": tts.QWEN_CLONE_TTS
        }], language="zh", tts_type=tts.QWEN_CLONE_TTS)
        wk.uito.connect(feed)
        wk.start()
        winobj.test.setText(tr("Testing..."))

    def save():
        _save_params()
        winobj.close()

    winobj.set.clicked.connect(save)
    winobj.test.clicked.connect(test)
    return winobj

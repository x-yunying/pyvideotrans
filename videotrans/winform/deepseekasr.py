def openwin():
    from pathlib import Path

    from videotrans import recognition
    from videotrans.configure.config import params, tr
    from videotrans.util.help_misc import show_error
    from videotrans.util.TestSTT import TestSTT
    from videotrans.winform import get_cls
    from videotrans.winform._helpers import make_feed_stt

    winobj = get_cls(Path(__file__).stem)()
    winobj.update_ui()

    feed = make_feed_stt(winobj, "test")

    def test():
        key = winobj.deepseek_key.text().strip()
        if not key:
            show_error(tr("Must fill in the API Key"))
            return
        params["deepseek_key"] = key
        params.save()
        winobj.test.setText(tr("Testing..."))
        task = TestSTT(parent=winobj, recogn_type=recognition.DEEPSEEK_ASR)
        task.uito.connect(feed)
        task.start()

    def save():
        key = winobj.deepseek_key.text().strip()
        if not key:
            show_error(tr("Must fill in the API Key"))
            return
        params["deepseek_key"] = key
        params.save()
        winobj.close()

    winobj.set.clicked.connect(save)
    winobj.test.clicked.connect(test)
    return winobj

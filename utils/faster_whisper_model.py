from faster_whisper import WhisperModel

def faster_whisper_test(filename):
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type='float16')
    segments, info = model.transcribe(filename, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
import time
from diarization import diarization, speaker_diarization, speech_to_text

def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(raw_segments):
    return "\n\n".join(
        [
            chunk["speaker"] + " " + tuple_to_string(chunk["timestamp"]) + chunk["text"]
            for chunk in raw_segments
        ]
    )

if __name__ == "__main__":
    start_time = time.time()
    # result = speaker_diarization()
    # segments = []
    # for segment, track, label in result[0].itertracks(yield_label=True):
    #     segments.append({'segment': {'start': segment.start, 'end': segment.end},
    #                         'track': track,
    #                         'label': label})
    # print(segments)
    

    result = diarization()
    print("Diarization result: ", format_as_transcription(result[0]))
    print("total time: ", time.time() - start_time)
    # print("Hello World")
    with open('download\output.txt', 'w') as file:
        file.write(format_as_transcription(result[0]))
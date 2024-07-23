from pyannote.audio import Pipeline
from transformers import pipeline
from speechbox import ASRDiarizationPipeline
import os
import torchaudio


AUTH_TOKEN = os.getenv("AUTH_TOKEN")


def get_audio_files(directory):
    project_dir =  os.path.dirname(os.path.abspath(__file__))
    download_folder = os.path.join(project_dir, directory)
    # Expanded list of recognized audio file extensions  
    audio_extensions = [  
        '.3gp',  '.aa',   '.aac',  '.aax',  '.act',  '.aiff', '.alac',  
        '.amr',  '.ape',  '.au',   '.awb',  '.dss',  '.dvf',  '.flac',  
        '.gsm',  '.iklax','.ivs',  '.m4a',  '.m4b',  '.m4p',  '.mmf',  
        '.mp3',  '.mpc',  '.msv',  '.nmf',  '.ogg',  '.oga',  '.mogg',  
        '.opus', '.ra',   '.rm',   '.raw',  '.sln',  '.tta',  '.vox',  
        '.wav',  '.wma',  '.wv',   '.webm', '.8svx', '.cda'  
    ]  

    # List to hold the paths of identified audio files  
    audio_files = []  

    # Traverse the directory  
    for root, _, files in os.walk(download_folder):  
        for file in files:  
            if any(file.lower().endswith(ext) for ext in audio_extensions):  
                audio_files.append(os.path.join(root, file))  

    return audio_files 

def diarization():
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=AUTH_TOKEN
    )
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium.en",
        chunk_length_s=30,
    )
    speech_pipeline = ASRDiarizationPipeline(
        asr_pipeline = asr_pipeline, diarization_pipeline = diarization_pipeline
    )
    diarization_result = []
    audio_file_list = get_audio_files('download')
    for audio_file in audio_file_list:
        diarization_result.append(speech_pipeline(audio_file))
    return diarization_result


def speaker_diarization():
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=AUTH_TOKEN
    )
    audio_file_list = get_audio_files('download')
    diarizations = []
    print(audio_file_list)
    for audio_file in audio_file_list:
        waveform, sample_rate = torchaudio.load(audio_file)
        diarizations.append(diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2))
    return diarizations

def speech_to_text():
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium.en",
        chunk_length_s=30,
    )
    audio_file_list = get_audio_files('download')
    text_result = asr_pipeline(
        audio_file_list[0],
        batch_size=8,
        return_timestamps=True,
    )["chunks"]
    print(text_result)


def audio_convert_to_wav(audio_file):  
    if audio_file.lower().endswith('.wav'):  
        return audio_file  
    else:  
        # Load the audio file using torchaudio  
        waveform, sample_rate = torchaudio.load(audio_file)  

        # Set the path for the output WAV file  
        wav_path = os.path.splitext(audio_file)[0] + ".wav"  

        # Save the waveform to a WAV file  
        torchaudio.save(wav_path, waveform, sample_rate)  

        return wav_path  
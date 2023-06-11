from bark.generation import load_codec_model, generate_text_semantic, grab_best_device
from bark import SAMPLE_RATE
from encodec.utils import convert_audio
from bark.hubert.hubert_manager import HuBERTManager
from bark.hubert.pre_kmeans_hubert import CustomHubert
from bark.hubert.customtokenizer import CustomTokenizer
from bark.api import semantic_to_waveform
from scipy.io.wavfile import write as write_wav
from util.helper import create_filename
from util.settings import Settings


import torchaudio
import torch
import os
import gradio

def swap_voice_from_audio(swap_audio_filename, selected_speaker, tokenizer_lang, seed, batchcount, progress=gradio.Progress(track_tqdm=True)):
    use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
    progress(0, desc="Loading Codec")
    
    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed(tokenizer_lang=tokenizer_lang)

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer 
    # Load HuBERT for semantic tokens

    # Load the HuBERT model
    device = grab_best_device(use_gpu)
    hubert_model = CustomHubert(checkpoint_path='./models/hubert/hubert.pt').to(device)
    model = load_codec_model(use_gpu=use_gpu)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint(f'./models/hubert/{tokenizer_lang}_tokenizer.pth').to(device)  # Automatically uses the right layers

    progress(0.25, desc="Converting WAV")

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(swap_audio_filename)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    audio = semantic_to_waveform(
        semantic_tokens,
        history_prompt=selected_speaker,
        temp=0.7,
        silent=False,
        output_full=False)

    settings = Settings('config.yaml')

    result = create_filename(settings.output_folder_path, None, "swapvoice",".wav")
    write_wav(result, SAMPLE_RATE, audio)
    return result


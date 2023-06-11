from bark.generation import load_codec_model, generate_text_semantic, grab_best_device
from encodec.utils import convert_audio
from bark.hubert.hubert_manager import HuBERTManager
from bark.hubert.pre_kmeans_hubert import CustomHubert
from bark.hubert.customtokenizer import CustomTokenizer

import torchaudio
import torch
import os
import gradio


def clone_voice(audio_filepath, dest_filename, progress=gradio.Progress(track_tqdm=True)):
    # if len(text) < 1:
    #    raise gradio.Error('No transcription text entered!')

    use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
    progress(0, desc="Loading Codec")
    model = load_codec_model(use_gpu=use_gpu)
    
    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer 
    # Load HuBERT for semantic tokens

    # Load the HuBERT model
    device = grab_best_device(use_gpu)
    hubert_model = CustomHubert(checkpoint_path='./models/hubert/hubert.pt').to(device)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint('./models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers

    progress(0.25, desc="Converting WAV")

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_filepath)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    progress(0.5, desc="Extracting codes")

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # get seconds of audio
    # seconds = wav.shape[-1] / model.sample_rate
    # generate semantic tokens
    # semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)

    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    import numpy as np
    output_path = dest_filename + '.npz'
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    return "Finished"

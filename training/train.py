import os
import fnmatch
import shutil

import numpy
import torchaudio
import gradio

from bark.hubert.pre_kmeans_hubert import CustomHubert
from bark.hubert.customtokenizer import auto_train
from tqdm.auto import tqdm


def training_prepare_files(path, model,progress=gradio.Progress(track_tqdm=True)):

    semanticsfolder = "./training/data/output"
    wavfolder = "./training/data/output_wav"
    ready = os.path.join(path, 'ready')

    testfiles = fnmatch.filter(os.listdir(ready), '*.npy')
    if(len(testfiles) < 1):
        # prepare and copy for training
        hubert_model = CustomHubert(checkpoint_path=model)

        wavfiles = fnmatch.filter(os.listdir(wavfolder), '*.wav')
        for i, f in tqdm(enumerate(wavfiles), total=len(wavfiles)):
            semaname = '.'.join(f.split('.')[:-1])  # Cut off the extension
            semaname = f'{semaname}.npy'
            semafilename =  os.path.join(semanticsfolder, semaname)
            if not os.path.isfile(semafilename):
                print(f'Skipping {f} no semantics pair found!')
                continue

            print('Processing', f)
            wav, sr = torchaudio.load(os.path.join(wavfolder, f))
            if wav.shape[0] == 2:  # Stereo to mono if needed
                wav = wav.mean(0, keepdim=True)
            output = hubert_model.forward(wav, input_sample_hz=sr)
            out_array = output.cpu().numpy()
            fname = f'{i}_semantic_features.npy'
            numpy.save(os.path.join(ready, fname), out_array)
            fname = f'{i}_semantic.npy'
            shutil.copy(semafilename, os.path.join(ready, fname))

def train(path, save_every, max_epochs):
    auto_train(path, save_epochs=save_every)


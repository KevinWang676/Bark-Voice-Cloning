import random
import uuid
import numpy
import os
import random
import fnmatch

from tqdm.auto import tqdm
from scipy.io import wavfile

from bark.generation import load_model, SAMPLE_RATE
from bark.api import semantic_to_waveform

from bark import text_to_semantic
from bark.generation import load_model

from training.data import load_books, random_split_chunk

output = 'training/data/output'
output_wav = 'training/data/output_wav'


def prepare_semantics_from_text(num_generations):
    loaded_data = load_books(True)

    print('Loading semantics model')
    load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')

    if not os.path.isdir(output):
        os.mkdir(output)

    loop = 1
    while 1:
        filename = uuid.uuid4().hex + '.npy'
        file_name = os.path.join(output, filename)
        text = ''
        while not len(text) > 0:
            text = random_split_chunk(loaded_data)  # Obtain a short chunk of text
            text = text.strip()
        print(f'{loop} Generating semantics for text:', text)
        loop+=1 
        semantics = text_to_semantic(text, temp=round(random.uniform(0.6, 0.8), ndigits=2))
        numpy.save(file_name, semantics)


def prepare_wavs_from_semantics():
    if not os.path.isdir(output):
        raise Exception('No \'output\' folder, make sure you run create_data.py first!')
    if not os.path.isdir(output_wav):
        os.mkdir(output_wav)

    print('Loading coarse model')
    load_model(use_gpu=True, use_small=False, force_reload=False, model_type='coarse')
    print('Loading fine model')
    load_model(use_gpu=True, use_small=False, force_reload=False, model_type='fine')

    files = fnmatch.filter(os.listdir(output), '*.npy')
    current = 1
    total = len(files)

    for i, f in tqdm(enumerate(files), total=len(files)):
        real_name = '.'.join(f.split('.')[:-1])  # Cut off the extension
        file_name = os.path.join(output, f)
        out_file = os.path.join(output_wav, f'{real_name}.wav')
        if not os.path.isfile(out_file) and os.path.isfile(file_name):  # Don't process files that have already been processed, to be able to continue previous generations
            print(f'Processing ({i+1}/{total}) -> {f}')
            wav = semantic_to_waveform(numpy.load(file_name), temp=round(random.uniform(0.6, 0.8), ndigits=2))
            # Change to PCM16
            # wav = (wav * 32767).astype(np.int16)
            wavfile.write(out_file, SAMPLE_RATE, wav)

    print('Done!')


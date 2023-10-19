from cProfile import label
import dataclasses
from distutils.command.check import check
from doctest import Example
import gradio as gr
import os
import sys
import numpy as np
import logging
import torch
import pytorch_seed
import time

from xml.sax import saxutils
from bark.api import generate_with_settings
from bark.api import save_as_prompt
from util.settings import Settings
#import nltk

from bark import SAMPLE_RATE
from cloning.clonevoice import clone_voice
from bark.generation import SAMPLE_RATE, preload_models, _load_history_prompt, codec_decode
from scipy.io.wavfile import write as write_wav
from util.parseinput import split_and_recombine_text, build_ssml, is_ssml, create_clips_from_ssml
from datetime import datetime
from tqdm.auto import tqdm
from util.helper import create_filename, add_id3_tag
from swap_voice import swap_voice_from_audio
from training.training_prepare import prepare_semantics_from_text, prepare_wavs_from_semantics
from training.train import training_prepare_files, train

settings = Settings('config.yaml')

def generate_text_to_speech(text, selected_speaker, text_temp, waveform_temp, eos_prob, quick_generation, complete_settings, seed, batchcount, progress=gr.Progress(track_tqdm=True)):
    # Chunk the text into smaller pieces then combine the generated audio

    # generation settings
    if selected_speaker == 'None':
        selected_speaker = None

    voice_name = selected_speaker

    if text == None or len(text) < 1:
       if selected_speaker == None:
            raise gr.Error('No text entered!')

       # Extract audio data from speaker if no text and speaker selected
       voicedata = _load_history_prompt(voice_name)
       audio_arr = codec_decode(voicedata["fine_prompt"])
       result = create_filename(settings.output_folder_path, "None", "extract",".wav")
       save_wav(audio_arr, result)
       return result

    if batchcount < 1:
        batchcount = 1


    silenceshort = np.zeros(int((float(settings.silence_sentence) / 1000.0) * SAMPLE_RATE), dtype=np.int16)  # quarter second of silence
    silencelong = np.zeros(int((float(settings.silence_speakers) / 1000.0) * SAMPLE_RATE), dtype=np.float32)  # half a second of silence
    use_last_generation_as_history = "Use last generation as history" in complete_settings
    save_last_generation = "Save generation as Voice" in complete_settings
    for l in range(batchcount):
        currentseed = seed
        if seed != None and seed > 2**32 - 1:
            logger.warning(f"Seed {seed} > 2**32 - 1 (max), setting to random")
            currentseed = None
        if currentseed == None or currentseed <= 0:
            currentseed = np.random.default_rng().integers(1, 2**32 - 1)
        assert(0 < currentseed and currentseed < 2**32)

        progress(0, desc="Generating")

        full_generation = None

        all_parts = []
        complete_text = ""
        text = text.lstrip()
        if is_ssml(text):
            list_speak = create_clips_from_ssml(text)
            prev_speaker = None
            for i, clip in tqdm(enumerate(list_speak), total=len(list_speak)):
                selected_speaker = clip[0]
                # Add pause break between speakers
                if i > 0 and selected_speaker != prev_speaker:
                    all_parts += [silencelong.copy()]
                prev_speaker = selected_speaker
                text = clip[1]
                text = saxutils.unescape(text)
                if selected_speaker == "None":
                    selected_speaker = None

                print(f"\nGenerating Text ({i+1}/{len(list_speak)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                complete_text += text
                with pytorch_seed.SavedRNG(currentseed):
                    audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)
                    currentseed = torch.random.initial_seed()
                if len(list_speak) > 1:
                    filename = create_filename(settings.output_folder_path, currentseed, "audioclip",".wav")
                    save_wav(audio_array, filename)
                    add_id3_tag(filename, text, selected_speaker, currentseed)

                all_parts += [audio_array]
        else:
            texts = split_and_recombine_text(text, settings.input_text_desired_length, settings.input_text_max_length)
            for i, text in tqdm(enumerate(texts), total=len(texts)):
                print(f"\nGenerating Text ({i+1}/{len(texts)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                complete_text += text
                if quick_generation == True:
                    with pytorch_seed.SavedRNG(currentseed):
                        audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)
                        currentseed = torch.random.initial_seed()
                else:
                    full_output = use_last_generation_as_history or save_last_generation
                    if full_output:
                        full_generation, audio_array = generate_with_settings(text_prompt=text, voice_name=voice_name, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob, output_full=True)
                    else:
                        audio_array = generate_with_settings(text_prompt=text, voice_name=voice_name, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)

                # Noticed this in the HF Demo - convert to 16bit int -32767/32767 - most used audio format
                # audio_array = (audio_array * 32767).astype(np.int16)

                if len(texts) > 1:
                    filename = create_filename(settings.output_folder_path, currentseed, "audioclip",".wav")
                    save_wav(audio_array, filename)
                    add_id3_tag(filename, text, selected_speaker, currentseed)

                if quick_generation == False and (save_last_generation == True or use_last_generation_as_history == True):
                    # save to npz
                    voice_name = create_filename(settings.output_folder_path, seed, "audioclip", ".npz")
                    save_as_prompt(voice_name, full_generation)
                    if use_last_generation_as_history:
                        selected_speaker = voice_name

                all_parts += [audio_array]
                # Add short pause between sentences
                if text[-1] in "!?.\n" and i > 1:
                    all_parts += [silenceshort.copy()]

        # save & play audio
        result = create_filename(settings.output_folder_path, currentseed, "final",".wav")
        save_wav(np.concatenate(all_parts), result)
        # write id3 tag with text truncated to 60 chars, as a precaution...
        add_id3_tag(result, complete_text, selected_speaker, currentseed)

    return result



def save_wav(audio_array, filename):
    write_wav(filename, SAMPLE_RATE, audio_array)

def save_voice(filename, semantic_prompt, coarse_prompt, fine_prompt):
    np.savez_compressed(
        filename,
        semantic_prompt=semantic_prompt,
        coarse_prompt=coarse_prompt,
        fine_prompt=fine_prompt
    )


def on_quick_gen_changed(checkbox):
    if checkbox == False:
        return gr.CheckboxGroup.update(visible=True)
    return gr.CheckboxGroup.update(visible=False)

def delete_output_files(checkbox_state):
    if checkbox_state:
        outputs_folder = os.path.join(os.getcwd(), settings.output_folder_path)
        if os.path.exists(outputs_folder):
            purgedir(outputs_folder)
    return False


# https://stackoverflow.com/a/54494779
def purgedir(parent):
    for root, dirs, files in os.walk(parent):
        for item in files:
            # Delete subordinate files
            filespec = os.path.join(root, item)
            os.unlink(filespec)
        for item in dirs:
            # Recursively perform this operation for subordinate directories
            purgedir(os.path.join(root, item))

def convert_text_to_ssml(text, selected_speaker):
    return build_ssml(text, selected_speaker)


def training_prepare(selected_step, num_text_generations, progress=gr.Progress(track_tqdm=True)):
    if selected_step == prepare_training_list[0]:
        prepare_semantics_from_text()
    else:
        prepare_wavs_from_semantics()
    return None


def start_training(save_model_epoch, max_epochs, progress=gr.Progress(track_tqdm=True)):
    training_prepare_files("./training/data/", "./training/data/checkpoint/hubert_base_ls960.pt")
    train("./training/data/", save_model_epoch, max_epochs)
    return None



def apply_settings(themes, input_server_name, input_server_port, input_server_public, input_desired_len, input_max_len, input_silence_break, input_silence_speaker):
    settings.selected_theme = themes
    settings.server_name = input_server_name
    settings.server_port = input_server_port
    settings.server_share = input_server_public
    settings.input_text_desired_length = input_desired_len
    settings.input_text_max_length = input_max_len
    settings.silence_sentence = input_silence_break
    settings.silence_speaker = input_silence_speaker
    settings.save()

def restart():
    global restart_server
    restart_server = True


def create_version_html():
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
 •
torch: {getattr(torch, '__long_version__',torch.__version__)}
 •
gradio: {gr.__version__}
"""
    return versions_html



logger = logging.getLogger(__name__)
APPTITLE = "Bark Voice Cloning UI"


autolaunch = False

if len(sys.argv) > 1:
    autolaunch = "-autolaunch" in sys.argv

if torch.cuda.is_available() == False:
    os.environ['BARK_FORCE_CPU'] = 'True'
    logger.warning("No CUDA detected, fallback to CPU!")

print(f'smallmodels={os.environ.get("SUNO_USE_SMALL_MODELS", False)}')
print(f'enablemps={os.environ.get("SUNO_ENABLE_MPS", False)}')
print(f'offloadcpu={os.environ.get("SUNO_OFFLOAD_CPU", False)}')
print(f'forcecpu={os.environ.get("BARK_FORCE_CPU", False)}')
print(f'autolaunch={autolaunch}\n\n')

#print("Updating nltk\n")
#nltk.download('punkt')

print("Preloading Models\n")
preload_models()

available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
tokenizer_language_list = ["de","en", "pl"]
prepare_training_list = ["Step 1: Semantics from Text","Step 2: WAV from Semantics"]

seed = -1
server_name = settings.server_name
if len(server_name) < 1:
    server_name = None
server_port = settings.server_port
if server_port <= 0:
    server_port = None
global run_server
global restart_server

run_server = True

while run_server:
    # Collect all existing speakers/voices in dir
    speakers_list = []

    for root, dirs, files in os.walk("./bark/assets/prompts"):
        for file in files:
            if file.endswith(".npz"):
                pathpart = root.replace("./bark/assets/prompts", "")
                name = os.path.join(pathpart, file[:-4])
                if name.startswith("/") or name.startswith("\\"):
                     name = name[1:]
                speakers_list.append(name)

    speakers_list = sorted(speakers_list, key=lambda x: x.lower())
    speakers_list.insert(0, 'None')

    print(f'Launching {APPTITLE} Server')

    # Create Gradio Blocks

    with gr.Blocks(title=f"{APPTITLE}", mode=f"{APPTITLE}", theme=settings.selected_theme) as barkgui:
        gr.Markdown("# <center>🐶🎶⭐ - Bark Voice Cloning</center>")
        gr.Markdown("## <center>🤗 - If you like this space, please star my [github repo](https://github.com/KevinWang676/Bark-Voice-Cloning)</center>")
        gr.Markdown("### <center>🎡 - Based on [bark-gui](https://github.com/C0untFloyd/bark-gui)</center>")
        gr.Markdown(f""" You can duplicate and use it with a GPU: <a href="https://huggingface.co/spaces/{os.getenv('SPACE_ID')}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a>
                         or open in [Colab](https://colab.research.google.com/github/KevinWang676/Bark-Voice-Cloning/blob/main/Bark_Voice_Cloning_UI.ipynb) for quick start 🌟
                    """)

        with gr.Tab("🎙️ - Clone Voice"):
            with gr.Row():
                input_audio_filename = gr.Audio(label="Input audio.wav", source="upload", type="filepath")
            #transcription_text = gr.Textbox(label="Transcription Text", lines=1, placeholder="Enter Text of your Audio Sample here...")
            with gr.Row():
                with gr.Column():
                    initialname = "/content/Bark-Voice-Cloning/bark/assets/prompts/file"
                    output_voice = gr.Textbox(label="Filename of trained Voice (do not change the initial name)", lines=1, placeholder=initialname, value=initialname, visible=False)
                with gr.Column():
                    tokenizerlang = gr.Dropdown(tokenizer_language_list, label="Base Language Tokenizer", value=tokenizer_language_list[1], visible=False)
            with gr.Row():
                clone_voice_button = gr.Button("Create Voice", variant="primary")
            with gr.Row():
                dummy = gr.Text(label="Progress")
                npz_file = gr.File(label=".npz file")
            speakers_list.insert(0, npz_file) # add prompt

        with gr.Tab("🎵 - TTS"):
            with gr.Row():
                with gr.Column():
                    placeholder = "Enter text here."
                    input_text = gr.Textbox(label="Input Text", lines=4, placeholder=placeholder)
                    convert_to_ssml_button = gr.Button("Convert Input Text to SSML")
                with gr.Column():
                        seedcomponent = gr.Number(label="Seed (default -1 = Random)", precision=0, value=-1)
                        batchcount = gr.Number(label="Batch count", precision=0, value=1)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("[Voice Prompt Library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)")
                    speaker = gr.Dropdown(speakers_list, value=speakers_list[0], label="Voice (Choose “file” if you wanna use the custom voice)")

                with gr.Column():
                    text_temp = gr.Slider(0.1, 1.0, value=0.6, label="Generation Temperature", info="1.0 more diverse, 0.1 more conservative")
                    waveform_temp = gr.Slider(0.1, 1.0, value=0.7, label="Waveform temperature", info="1.0 more diverse, 0.1 more conservative")

            with gr.Row():
                with gr.Column():
                    quick_gen_checkbox = gr.Checkbox(label="Quick Generation", value=True)
                    settings_checkboxes = ["Use last generation as history", "Save generation as Voice"]
                    complete_settings = gr.CheckboxGroup(choices=settings_checkboxes, value=settings_checkboxes, label="Detailed Generation Settings", type="value", interactive=True, visible=False)
                with gr.Column():
                    eos_prob = gr.Slider(0.0, 0.5, value=0.05, label="End of sentence probability")

            with gr.Row():
                with gr.Column():
                    tts_create_button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    hidden_checkbox = gr.Checkbox(visible=False)
                    button_stop_generation = gr.Button("Stop generation")
            with gr.Row():
                output_audio = gr.Audio(label="Generated Audio", type="filepath")

        with gr.Tab("🔮 - Voice Conversion"):
            with gr.Row():
                 swap_audio_filename = gr.Audio(label="Input audio.wav to swap voice", source="upload", type="filepath")
            with gr.Row():
                 with gr.Column():
                     swap_tokenizer_lang = gr.Dropdown(tokenizer_language_list, label="Base Language Tokenizer", value=tokenizer_language_list[1])
                     swap_seed = gr.Number(label="Seed (default -1 = Random)", precision=0, value=-1)
                 with gr.Column():
                     speaker_swap = gr.Dropdown(speakers_list, value=speakers_list[0], label="Voice (Choose “file” if you wanna use the custom voice)")
                     swap_batchcount = gr.Number(label="Batch count", precision=0, value=1)
            with gr.Row():
                swap_voice_button = gr.Button("Generate", variant="primary")
            with gr.Row():
                output_swap = gr.Audio(label="Generated Audio", type="filepath")


        quick_gen_checkbox.change(fn=on_quick_gen_changed, inputs=quick_gen_checkbox, outputs=complete_settings)
        convert_to_ssml_button.click(convert_text_to_ssml, inputs=[input_text, speaker],outputs=input_text)
        gen_click = tts_create_button.click(generate_text_to_speech, inputs=[input_text, speaker, text_temp, waveform_temp, eos_prob, quick_gen_checkbox, complete_settings, seedcomponent, batchcount],outputs=output_audio)
        button_stop_generation.click(fn=None, inputs=None, outputs=None, cancels=[gen_click])



        swap_voice_button.click(swap_voice_from_audio, inputs=[swap_audio_filename, speaker_swap, swap_tokenizer_lang, swap_seed, swap_batchcount], outputs=output_swap)
        clone_voice_button.click(clone_voice, inputs=[input_audio_filename, output_voice], outputs=[dummy, npz_file])


        restart_server = False
        try:
            barkgui.queue().launch(show_error=True, share=True)
        except:
            restart_server = True
            run_server = False
        try:
            while restart_server == False:
                time.sleep(1.0)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        barkgui.close()

import os
from datetime import datetime
from mutagen.wave import WAVE
from mutagen.id3._frames import *

def create_filename(path, seed, name, extension):
    now = datetime.now()
    date_str =now.strftime("%m-%d-%Y")
    outputs_folder = os.path.join(os.getcwd(), path)
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_folder = os.path.join(outputs_folder, date_str)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    time_str = now.strftime("%H-%M-%S")
    if seed == None:
        file_name = f"{name}_{time_str}{extension}"
    else:
        file_name = f"{name}_{time_str}_s{seed}{extension}"
    return os.path.join(sub_folder, file_name)


def add_id3_tag(filename, text, speakername, seed):
    audio = WAVE(filename)
    if speakername == None:
        speakername = "Unconditional"

    # write id3 tag with text truncated to 60 chars, as a precaution...
    audio["TIT2"] = TIT2(encoding=3, text=text[:60])
    audio["TPE1"] = TPE1(encoding=3, text=f"Voice {speakername} using Seed={seed}")
    audio["TPUB"] = TPUB(encoding=3, text="Bark by Suno AI")
    audio["COMMENT"] = COMM(encoding=3, text="Generated with Bark GUI - Text-Prompted Generative Audio Model. Visit https://github.com/C0untFloyd/bark-gui")
    audio.save()

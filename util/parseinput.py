import re
import xml.etree.ElementTree as ET
from xml.sax import saxutils
#import nltk

# Chunked generation originally from https://github.com/serp-ai/bark-with-voice-clone
def split_and_recombine_text(text, desired_length=100, max_length=150):
    # return nltk.sent_tokenize(text)

    # from https://github.com/neonbjb/tortoise-tts
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in "!?.,\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?]\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?.]"
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv

def is_ssml(value):
    try:
        ET.fromstring(value)
    except ET.ParseError:
        return False
    return True

def build_ssml(rawtext, selected_voice):
    texts = rawtext.split("\n")
    joinedparts = ""
    for textpart in texts:
        textpart = textpart.strip()
        if len(textpart) < 1:
            continue
        joinedparts = joinedparts + f"\n<voice name=\"{selected_voice}\">{saxutils.escape(textpart)}</voice>"
    ssml = f"""<?xml version="1.0"?>
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                   http://www.w3.org/TR/speech-synthesis/synthesis.xsd"
         xml:lang="en-US">
         {joinedparts}
</speak>
    """
    return ssml

def create_clips_from_ssml(ssmlinput):
    # Parse the XML
    tree = ET.ElementTree(ET.fromstring(ssmlinput))
    root = tree.getroot()

    # Create an empty list
    voice_list = []

    # Loop through all voice tags
    for voice in root.iter('{http://www.w3.org/2001/10/synthesis}voice'):
        # Extract the voice name attribute and the content text
        voice_name = voice.attrib['name']
        voice_content = voice.text.strip() if voice.text else ''
        if(len(voice_content) > 0):
            parts = split_and_recombine_text(voice_content)
            for p in parts:
                if(len(p) > 1):
                    # add to tuple list
                    voice_list.append((voice_name, p))
    return voice_list


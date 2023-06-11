import os.path
import shutil
import urllib.request

import huggingface_hub


class HuBERTManager:


    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', file_name: str = 'hubert.pt'):
        install_dir = os.path.join('models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, file_name)
        if not os.path.isfile(install_file):
            print(f'Downloading HuBERT base model from {download_url}')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')
        return install_file


    @staticmethod
    def make_sure_tokenizer_installed(model: str = 'quantifier_hubert_base_ls960_14.pth', repo: str = 'GitMylo/bark-voice-cloning', tokenizer_lang: str = 'en'):
        local_file = tokenizer_lang + '_tokenizer.pth'
        install_dir = os.path.join('models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, local_file)
        if not os.path.isfile(install_file):
            # refactor to use lists
            if tokenizer_lang == 'en':
                repo = 'GitMylo/bark-voice-cloning'
                model = 'quantifier_hubert_base_ls960_14.pth'
            elif tokenizer_lang == 'de':
                repo = 'CountFloyd/bark-voice-cloning-german-HuBERT-quantizer'
                model = 'german-HuBERT-quantizer_14_epoch.pth'
            elif tokenizer_lang == 'pl':
                repo = 'Hobis/bark-voice-cloning-polish-HuBERT-quantizer'
                model = 'polish-HuBERT-quantizer_8_epoch.pth'
            else:
                raise 'Unknown Tokenizer Language!'
            print(f'{local_file} not found. Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, model, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, model), install_file)
            print('Downloaded tokenizer')
        return install_file

import yaml

class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            self.selected_theme = data.get('selected_theme', "gstaff/xkcd")
            self.server_name = data.get('server_name', "")
            self.server_port = data.get('server_port', 0)
            self.server_share = data.get('server_share', False)
            self.input_text_desired_length = data.get('input_text_desired_length', 110)
            self.input_text_max_length = data.get('input_text_max_length', 170)
            self.silence_sentence = data.get('silence_between_sentences', 250)
            self.silence_speakers = data.get('silence_between_speakers', 500)
            self.output_folder_path = data.get('output_folder_path', 'outputs')

        except:
            self.selected_theme = "gstaff/xkcd"

    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'input_text_desired_length' : self.input_text_desired_length,
            'input_text_max_length' : self.input_text_max_length, 
            'silence_between_sentences': self.silence_sentence,
            'silence_between_speakers': self.silence_speakers,
            'output_folder_path': self.output_folder_path
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)




import os
import subprocess
import config
import utils
import platform
import requests
import shutil
import logging


try:
    import torch
except ModuleNotFoundError:
    logging.info("The torch module for Silero TTS is not found. Please run 'pip install -r silero_tts_requirements.txt' to install the required packages.")
    raise

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=config.LOGGING_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


class SileroTTSClient:
    def __init__(self, verbose=False):
        """Initialize the Piper TTS client."""
        device = torch.device(config.SILERO_DEVICE)
        torch.set_num_threads(config.SILERO_CPU_THREADS)
        local_file = config.SILERO_FILE #'v3_1_ru.pt'

        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/ru/{local_file}',
                                   local_file)  

        self.model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        self.model.to(device)
        self.sample_rate = 24000
        self.speaker= config.SILERO_VOICE
        self.verbose = verbose

    def tts(self, text_to_speak, output_file):
        """
        This function uses the Silero TTS engine to convert text to speech.
        
        Args:
            text_to_speak (str): The text to be converted to speech.
            output_file (str): The path where the output audio file will be saved.
            
        Returns:
            str: "success" if the TTS process was successful, "failed" otherwise.
        """
        # Sanitize the text to be spoken
        text_to_speak = utils.sanitize_text(text_to_speak)

        # If there's no text left after sanitization, return "failed"
        if not text_to_speak.strip():
            logging.info("No text to speak after sanitization.")
            return "failed"

        # Determine the operating system

        try:
            audio_paths = self.model.save_wav(text=text_to_speak,
                                              speaker=self.speaker,
                                              sample_rate=self.sample_rate,
                                              put_accent=True,
                                              put_yo=True)
            shutil.copyfile(audio_paths, output_file)
            return "success"
        except requests.RequestException as e:
            logging.exception(f"Error calling Silero TSS: {e}")
            return "failed"
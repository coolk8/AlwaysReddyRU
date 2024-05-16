import config
import utils
import requests
import logging

try:
    from vosk_tts import Model, Synth
except ModuleNotFoundError:
    logging.info("The Vosk TTS module is not found. Please run 'pip install -r vosk_tts_requirements.txt' to install the required packages.")
    raise

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=config.LOGGING_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


class VoskTTSClient:
    def __init__(self, verbose=False):
        """Initialize the Vosk TTS client."""
        self.verbose = verbose
        self.model = Model(model_name=config.VOSK_MODEL)
        self.synth = Synth(self.model)
        self.speaker_id = config.VOSK_SPEAKER_ID

    def tts(self, text_to_speak, output_file):
        """
        This function uses the Vosk TTS engine to convert text to speech.
        
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
            self.synth.synth(text_to_speak, output_file, speaker_id=self.speaker_id)
            return "success"
        except requests.RequestException as e:
            logging.exception(f"Error calling VOSK TTS API: {e}")
            return "failed"
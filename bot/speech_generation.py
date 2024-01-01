import numpy as np
import torch
from abc import ABC, abstractmethod


class SpeechGeneration(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, text=None, sample_rate=48000) -> (int, np.array):
        pass


class SpeechGenerationV1(SpeechGeneration):
    """
    Использую дефолтный пример из https://github.com/snakers4/silero-models
    """

    def __init__(self) -> None:
        self.model = None

        self.load()

    def load(self):
        tts_language = "ru"
        tts_model_id = "v4_ru"
        device = torch.device("cpu")

        model_tts, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=tts_language,
            speaker=tts_model_id,
        )
        model_tts.to(device)

        self.model = model_tts

    def generate(self, text=None, sample_rate=48000) -> (int, np.array):
        speaker = "xenia"
        put_accent = True
        put_yo = False
        audio = self.model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
        )
        return sample_rate, audio

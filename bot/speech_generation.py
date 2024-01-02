import numpy as np
import torch
from abc import ABC, abstractmethod
import nltk

# from ruaccent import RUAccent
from tqdm import tqdm


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


class SpeechGenerationV2(SpeechGeneration):
    """
    Добавил ударения к SpeechGenerationV1
    """

    def __init__(self) -> None:
        self.model = None
        # self.accent_model = None

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

        # accentizer = RUAccent()
        # accentizer.load(
        #     omograph_model_size="big_poetry",
        #     use_dictionary=True,
        # )
        # text = "на двери висит замок."
        # print(accentizer.process_all(text))

        self.model = model_tts
        # self.accent_model = accentizer

    def generate(self, text=None, sample_rate=48000) -> (int, np.array):
        text_tokenized = nltk.sent_tokenize(text=text)

        speaker = "xenia"
        put_accent = True
        put_yo = True
        all_speech = torch.tensor([], dtype=torch.float32)
        print(text)
        for text_chunk in tqdm(text_tokenized):
            # text_chunk = self.accent_model.process_all(text_chunk)
            print(text_chunk)
            if len(text_chunk) > 3:
                audio = self.model.apply_tts(
                    text=text_chunk,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo,
                )
                all_speech = torch.cat([all_speech, audio])

        return sample_rate, all_speech

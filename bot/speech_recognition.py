import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import numpy as np
import whisperx
import librosa
from abc import ABC, abstractmethod


class SpeechRecognition(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(
        self,
        y: np.array,
        sampling_rate: int = None,
        resample: bool = True,
    ):
        pass


class SpeechRecognitionV1(SpeechRecognition):
    """
    Использую дефолтный пример из https://huggingface.co/openai/whisper-large-v3
    """

    def __init__(self) -> None:
        self.model = None
        self.load()

    def load(self):
        device = "cuda"
        torch_dtype = torch.float16

        whisper_id = "openai/whisper-large-v3"

        whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            use_flash_attention_2=True,
        )
        whisper.to(device)

        processor = AutoProcessor.from_pretrained(whisper_id)

        speech_recognition = pipeline(
            "automatic-speech-recognition",
            model=whisper,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )

        dataset = load_dataset(
            "distil-whisper/librispeech_long", "clean", split="validation"
        )
        sample = dataset[0]["audio"]
        print(sample)
        result = speech_recognition(sample)
        print(result["text"])

        self.model = speech_recognition

    def generate(
        self,
        y: np.array,
        sampling_rate: int = None,
        resample: bool = True,
    ):
        result = self.model({"sampling_rate": sampling_rate, "raw": y})
        speech_text = result["text"]
        return speech_text


class SpeechRecognitionV2(SpeechRecognition):
    """
    Использую дефолтный пример из https://github.com/m-bain/whisperX
    """

    def __init__(self) -> None:
        self.model = None
        self.sampling_rate = 16_000
        self.load()

    def load(self):
        device = "cuda"
        # compute_type = "float16"

        # (may reduce accuracy)
        compute_type = "int8"

        speech_recognition = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            vad_model=None,
        )
        self.model = speech_recognition

        dataset = load_dataset(
            "distil-whisper/librispeech_long", "clean", split="validation"
        )
        sample = dataset[0]["audio"]

        result = self.generate(
            y=sample["array"],
            sampling_rate=sample["sampling_rate"],
            resample=True,
        )
        print(result)

    def generate(
        self,
        y: np.array,
        sampling_rate: int = None,
        resample: bool = True,
    ):
        data_16k = y
        if resample:
            data_16k = librosa.resample(
                y=data_16k,
                orig_sr=sampling_rate,
                target_sr=self.sampling_rate,
            )

        speech_text = self.model.transcribe(
            np.array(data_16k, dtype=np.float32),
            batch_size=2,
        )
        if len(speech_text["segments"]) > 0:
            speech_text = " ".join([item["text"] for item in speech_text["segments"]])
        else:
            speech_text = ""

        return speech_text

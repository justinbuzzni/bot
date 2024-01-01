import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import numpy as np


class SpeechRecognitionV1:
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

    def __call__(self, y: np.array, sampling_rate: int):
        result = self.model({"sampling_rate": sampling_rate, "raw": y})
        speech_text = result["text"]
        return speech_text

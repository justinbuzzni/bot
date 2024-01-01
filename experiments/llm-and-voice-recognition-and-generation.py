import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import gradio as gr
import numpy as np
from functools import partial
import requests

from fastapi import FastAPI

CUSTOM_PATH = "/"

app = FastAPI()


# @app.get("/")
# def read_main():
#     return {"message": "This is your main app"}


def llm_chat(user_text=None):
    initial_prompt = f"GPT4 Correct User: Ты бот-помощник. Отвечай коротко и по существу.\n\n{user_text}<|end_of_turn|>GPT4 Correct Assistant:\n"
    result = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": initial_prompt,
            "temperature": 0.5,
            "max_tokens": 1024,
        },
    ).json()["text"][0]
    result = result[len(initial_prompt) :]
    return result


def generate_voice(
    text=None,
    sample_rate=48000,
    model_tts=None,
):
    speaker = "xenia"
    put_accent = True
    put_yo = False
    audio = model_tts.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate,
        put_accent=put_accent,
        put_yo=put_yo,
    )
    return sample_rate, audio


def transcribe(
    audio,
    speech_recognition=None,
    speech_generation=None,
):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    sample = {"sampling_rate": sr, "raw": y}
    print(sample)
    result = speech_recognition(sample)
    speech_text = result["text"]
    llm_response = llm_chat(user_text=speech_text)
    sample_rate, generated_audio = speech_generation(
        text=llm_response,
    )
    return [
        gr.Markdown(
            f"**Распознанная речь:**\n{speech_text}\n\n**Сгенерированный ответ:**\n{llm_response}"
        ),
        gr.Audio(
            value=(sample_rate, generated_audio.numpy()),
            autoplay=True,
        ),
    ]


# if __name__ == "__main__":
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

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
print(sample)
result = speech_recognition(sample)
print(result["text"])

import torch

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

markdown = gr.Markdown()
audio = gr.Audio()
demo = gr.Interface(
    partial(
        transcribe,
        speech_recognition=speech_recognition,
        speech_generation=partial(
            generate_voice,
            model_tts=model_tts,
        ),
    ),
    inputs=gr.Audio(sources=["microphone"]),
    outputs=[markdown, audio],
)
try:
    demo = gr.mount_gradio_app(
        app,
        demo.queue(),
        path=CUSTOM_PATH,
    )
    # demo.queue().launch(
    #     # inbrowser=True,
    #     server_port=7860,
    #     # share=True,
    # )

except KeyboardInterrupt:
    demo.close()
except Exception as e:
    print(e)
    demo.close()

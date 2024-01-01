import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import gradio as gr
import numpy as np
from functools import partial
import requests


def chat(user_text=None):
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


def transcribe(audio, pipe=None):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    sample = {"sampling_rate": sr, "raw": y}
    print(sample)
    result = pipe(sample)
    speech_text = result["text"]
    llm_response = chat(user_text=speech_text)
    return f"{speech_text}-----{llm_response}"


if __name__ == "__main__":
    device = "cuda"
    torch_dtype = torch.float16

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        use_flash_attention_2=True,
    )
    # model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
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
    result = pipe(sample)
    print(result["text"])

    demo = gr.Interface(
        partial(transcribe, pipe=pipe),
        gr.Audio(sources=["microphone"]),
        "text",
    )
    try:
        demo.launch(
            inbrowser=True,
            server_port=7866,
        )

        print("Launched on http://127.0.0.1:7866/")
    except KeyboardInterrupt:
        demo.close()
    except Exception as e:
        print(e)
        demo.close()

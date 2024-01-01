import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import gradio as gr
import numpy as np
from functools import partial

def transcribe(audio, pipe=None):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    sample = {"sampling_rate": sr, "raw": y}
    print(sample)
    result = pipe(sample)
    result = result["text"]
    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        use_flash_attention_2=True,
    )
    model.to(device)

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

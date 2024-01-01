import requests


def llm_chat_v1(user_text=None):
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

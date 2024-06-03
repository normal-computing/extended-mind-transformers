import anthropic
import requests

def query_claude(messages, api_key, model_name, max_tokens=100, temperature=0.7):
    client = anthropic.Client(api_key=api_key)
    system_prompt, messages = messages
    response = client.messages.create(
        model=model_name,
        system=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return response.content[0].text

def query_gpt(
    messages,
    api_key,
    model_name,
    max_tokens=100,
    temperature=0.7,
):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error in API request: {response.status_code}")
        print(response.text)
        return None

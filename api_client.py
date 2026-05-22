"""
LLM API client used to (a) generate gold-guided CoT rationales for the
distillation training set, and (b) call the LLM baselines (GLM-5,
DeepSeek-v4, ...) for the zero/few-shot rows in the main results table.

The client uses the standard OpenAI Chat Completions schema and any
provider that exposes an OpenAI-compatible endpoint can be plugged in
by setting the following environment variables:

    LLM_API_BASE   e.g. "https://api.openai.com/v1/chat/completions"
                       or any OpenAI-compatible gateway
    LLM_API_KEY    your bearer token

For mapping the short model keys we use throughout the code base
("glm5", "deepseek", ...) to the model id your provider expects, edit
the ``MODELS`` dict below.

Features:
  - Automatic retries with exponential back-off
  - Resume-on-restart via output JSONL: rows whose `id` already exists
    in the output file are skipped on the next run.
"""

import json
import os
import time
import requests
from typing import Optional


API_URL  = os.environ.get(
    "LLM_API_BASE",
    "https://api.openai.com/v1/chat/completions",
)
API_KEY  = os.environ.get("LLM_API_KEY", "")


# ----------------------------------------------------------------------
# Map the short keys used inside the code base to the concrete model id
# expected by your provider.  The defaults below assume an OpenAI-style
# gateway; change them to whatever your endpoint exposes.
# ----------------------------------------------------------------------
MODELS = {
    "glm5":     os.environ.get("LLM_MODEL_GLM5",     "glm-5"),
    "qwen":     os.environ.get("LLM_MODEL_QWEN",     "qwen-plus"),
    "deepseek": os.environ.get("LLM_MODEL_DEEPSEEK", "deepseek-chat"),
    "kimi":     os.environ.get("LLM_MODEL_KIMI",     "moonshot-v1-32k"),
    "minimax":  os.environ.get("LLM_MODEL_MINIMAX",  "abab6.5-chat"),
}


def call_api(
    model_key: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 120,
    max_retries: int = 3,
) -> Optional[dict]:
    """Call the chat-completions endpoint with retries.

    Returns the parsed JSON response, or ``None`` if all retries fail.
    """
    if not API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is not set. Export it before running, e.g.:\n"
            "    export LLM_API_KEY=sk-...\n"
            "    export LLM_API_BASE=https://api.openai.com/v1/chat/completions"
        )

    model_id = MODELS[model_key]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"  [Retry {attempt+1}/{max_retries}] {model_key} error: {e}, waiting {wait}s...")
            time.sleep(wait)
    return None


def extract_content(response: dict) -> str:
    """Return the assistant message content. Falls back to reasoning_content
    (used by some reasoning-model providers) when content is empty."""
    if not response:
        return ""
    try:
        msg = response["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning_content") or ""
        return content
    except (KeyError, IndexError):
        return ""


def extract_reasoning(response: dict) -> str:
    """Return the reasoning_content field when present (some providers
    expose intermediate chain-of-thought separately from the answer)."""
    if not response:
        return ""
    try:
        return response["choices"][0]["message"].get("reasoning_content", "")
    except (KeyError, IndexError):
        return ""


def batch_call(
    model_key: str,
    prompts: list[dict],
    output_path: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    delay: float = 0.5,
):
    """Call the API for a batch of prompts and stream JSONL to disk.

    ``prompts`` is a list of dicts of the form ``{"id": "...", "content": "..."}``.
    The function is resumable: rows whose ``id`` is already present in
    ``output_path`` are skipped on subsequent runs.
    """
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  [Resume] {len(done_ids)}/{len(prompts)} already done")

    remaining = [p for p in prompts if p["id"] not in done_ids]
    if not remaining:
        print(f"  [Skip] {model_key}: nothing to do")
        return

    print(f"  [Start] {model_key}: {len(remaining)} prompts")

    with open(output_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(remaining):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt["content"]})

            resp = call_api(model_key, messages, temperature=temperature, max_tokens=max_tokens)
            content = extract_content(resp)
            reasoning = extract_reasoning(resp)

            result = {
                "id": prompt["id"],
                "model": model_key,
                "content": content,
                "reasoning": reasoning,
            }
            for k, v in prompt.items():
                if k not in ("id", "content"):
                    result[k] = v

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{model_key}] {i+1}/{len(remaining)} done")

            time.sleep(delay)

    print(f"  [Done] {model_key}: wrote {output_path}")


if __name__ == "__main__":
    # Smoke test all configured models.
    test_msg = [{"role": "user", "content": "Say 'OK' if you can read this."}]
    for key in MODELS:
        print(f"Testing {key} ({MODELS[key]})...", end=" ")
        resp = call_api(key, test_msg, max_tokens=50, timeout=60)
        if resp:
            text = extract_content(resp) or ""
            reasoning = extract_reasoning(resp) or ""
            display = text[:60] if text else reasoning[:60]
            print(f"OK - {display}")
        else:
            print("FAILED")

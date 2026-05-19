"""LLM providers with streaming. Each ``stream_*`` is a generator of text deltas."""

from __future__ import annotations

import json
import os
from typing import Generator, Iterable

import requests

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 60))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

PROVIDER_NO_API = "Simple Extraction (No API)"
PROVIDER_OLLAMA = "Ollama (Local)"
PROVIDER_GROQ = "Groq (FREE)"
PROVIDER_HF = "HuggingFace (FREE)"
PROVIDER_TOGETHER = "Together AI (FREE)"

ALL_PROVIDERS = [
    PROVIDER_NO_API,
    PROVIDER_OLLAMA,
    PROVIDER_GROQ,
    PROVIDER_HF,
    PROVIDER_TOGETHER,
]

PROVIDER_ENV_VARS = {
    PROVIDER_GROQ: "GROQ_API_KEY",
    PROVIDER_HF: "HUGGINGFACE_API_KEY",
    PROVIDER_TOGETHER: "TOGETHER_API_KEY",
}

PROVIDER_SIGNUP_URLS = {
    PROVIDER_GROQ: "https://console.groq.com",
    PROVIDER_HF: "https://huggingface.co/settings/tokens",
    PROVIDER_TOGETHER: "https://api.together.xyz",
    PROVIDER_OLLAMA: "https://ollama.com  (then run: ollama pull llama3.2)",
}


def _build_messages(prompt: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": f"Answer based strictly on this context:\n\n{context}"},
        {"role": "user", "content": prompt},
    ]


def _stream_openai_compatible(url: str, headers: dict, payload: dict) -> Generator[str, None, None]:
    """Parse OpenAI-style server-sent events into text deltas."""
    payload = {**payload, "stream": True}
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta


def stream_groq(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    yield from _stream_openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": "llama3-8b-8192",
            "messages": _build_messages(prompt, context),
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )


def stream_together(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    yield from _stream_openai_compatible(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": _build_messages(prompt, context),
            "max_tokens": 1024,
        },
    )


def stream_huggingface(prompt: str, context: str, api_key: str) -> Generator[str, None, None]:
    # HF Inference API isn't reliably streamable on the public endpoint —
    # emit the whole response as a single chunk so the UI stays consistent.
    resp = requests.post(
        "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "inputs": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
            "parameters": {"max_new_tokens": 512, "temperature": 0.7},
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list) and result:
        yield result[0].get("generated_text", "").split("Answer:")[-1].strip()
    else:
        yield str(result)


def stream_ollama(
    prompt: str,
    context: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> Generator[str, None, None]:
    with requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": (
                "Use the context below to answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
            ),
            "stream": True,
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            piece = obj.get("response", "")
            if piece:
                yield piece
            if obj.get("done"):
                return


def simple_local_answer(prompt: str, context: str) -> str:
    """No-API extractive fallback."""
    sentences = context.split(".")
    keywords = [w.lower() for w in prompt.split() if len(w) > 2]
    relevant = [s.strip() for s in sentences if any(k in s.lower() for k in keywords)]
    if relevant:
        return "**Based on the documents:**\n\n" + ". ".join(relevant[:3]) + "."
    return "I found relevant context but couldn't generate a detailed answer. Please check the sources below."


def stream_answer(provider: str, prompt: str, context: str, api_key: str) -> Iterable[str]:
    """Unified dispatch. Always yields strings; falls back to extractive if no key."""
    if provider == PROVIDER_OLLAMA:
        yield from stream_ollama(prompt, context)
        return
    if provider == PROVIDER_GROQ and api_key:
        yield from stream_groq(prompt, context, api_key)
        return
    if provider == PROVIDER_TOGETHER and api_key:
        yield from stream_together(prompt, context, api_key)
        return
    if provider == PROVIDER_HF and api_key:
        yield from stream_huggingface(prompt, context, api_key)
        return
    yield simple_local_answer(prompt, context)


def resolve_api_key(provider: str, ui_input: str) -> str:
    """Prefer the sidebar input; fall back to the provider's env var."""
    if ui_input:
        return ui_input
    env_var = PROVIDER_ENV_VARS.get(provider)
    return os.environ.get(env_var, "") if env_var else ""

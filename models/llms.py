import os
import re
import time
import numpy as np
import httpx
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

_client = None

# Hard cap per request: 90s read, 10s connect.
# Without this, a hung connection waits 600s (default) before raising
# APITimeoutError, burning an entire retry slot and 10 minutes per hang.
_TIMEOUT = httpx.Timeout(timeout=90.0, connect=10.0)

MODEL = "gpt-4-turbo"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = OpenAI(api_key=api_key, timeout=_TIMEOUT)
    return _client


def _gpt4_completion(
    input_str: str,
    steps: int,
    settings,
    num_samples: int = 1,
    temp: float = 1.0,
    **kwargs,
) -> list[str]:
    """
    Query GPT-4-Turbo to continue a serialised time series.
    Returns a list of raw completion strings (one per sample).

    Matches the original LLMTime paper (Gruver et al. 2023) invocation:
    - Single API call with n=num_samples (not a per-sample loop)
    - Dynamic max_tokens from actual token density of the input
    - Prompt format with explicit "Sequence:\n" task instruction
    """
    import tiktoken
    client = _get_client()
    top_p = kwargs.get("top_p", 1.0)
    model = kwargs.get("gpt_model", MODEL)

    # Dynamic token budget: measure actual tokens-per-step from the input string
    # so the budget scales with the serialisation format (matches original paper)
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    n_input_tokens = len(enc.encode(input_str))
    n_input_steps  = max(len(input_str.split(settings.time_sep)), 1)
    avg_tokens_per_step = n_input_tokens / n_input_steps
    max_tokens = max(64, int(avg_tokens_per_step * steps))

    # Prompt format from original paper (Gruver et al. 2023, gpt.py)
    sys_msg = (
        "You are a helpful assistant that performs time series predictions. "
        "The user will provide a sequence and you will predict the remaining sequence. "
        "The sequence is represented by decimal strings separated by commas."
    )
    extra_input = (
        "Please continue the following sequence without producing any additional text. "
        "Do not say anything like 'the next terms in the sequence are', just return the numbers. "
        "Sequence:\n"
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": extra_input + input_str + settings.time_sep},
    ]

    # Single API call with n=num_samples — matches original paper and eliminates
    # the per-sample loop + inter-sample sleeps that inflated wall-clock time 10×
    for attempt in range(10):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                n=num_samples,
            )
            return [choice.message.content.strip() for choice in response.choices]
        except RateLimitError as e:
            msg     = str(e)
            wait_ms = re.search(r"try again in (\d+)ms", msg)
            wait_s  = re.search(r"try again in ([\d.]+)s",  msg)
            if wait_ms:
                wait = int(wait_ms.group(1)) / 1000.0 + 2.0
            elif wait_s:
                wait = float(wait_s.group(1)) + 2.0
            else:
                wait = 65.0
            print(f"\n  [Rate limit] attempt {attempt+1}: sleeping {wait:.1f}s ...")
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            wait = 10.0 * (attempt + 1)
            print(f"\n  [Timeout/ConnErr] attempt {attempt+1}: "
                  f"{type(e).__name__}, sleeping {wait:.0f}s ...")
            time.sleep(wait)

    raise RuntimeError("GPT-4-Turbo API failed after 10 retries.")


_mistral_client = None


def _get_mistral_client() -> OpenAI:
    """Mistral via their OpenAI-compatible endpoint."""
    global _mistral_client
    if _mistral_client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set. Add it to .env.")
        _mistral_client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            timeout=_TIMEOUT,
        )
    return _mistral_client


def _mistral_completion(
    input_str: str,
    steps: int,
    settings,
    num_samples: int = 1,
    temp: float = 1.0,
    **kwargs,
) -> list[str]:
    """
    Query Mistral Small via their OpenAI-compatible API.
    Identical prompt format to _gpt4_completion (Gruver et al. 2023).
    Uses GPT-3.5 tokenizer as a proxy for token-budget estimation
    (same approach as the original LLMTime Mistral implementation).
    """
    import tiktoken
    client = _get_mistral_client()
    model  = kwargs.get("mistral_model", "mistral-small-latest")
    top_p  = kwargs.get("top_p", 1.0)

    # Approximate token budget using GPT-3.5 tokenizer (Mistral is similar)
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    n_input_tokens = len(enc.encode(input_str))
    n_input_steps  = max(len(input_str.split(settings.time_sep)), 1)
    avg_tokens_per_step = n_input_tokens / n_input_steps
    max_tokens = max(64, int(avg_tokens_per_step * steps))

    sys_msg = (
        "You are a helpful assistant that performs time series predictions. "
        "The user will provide a sequence and you will predict the remaining sequence. "
        "The sequence is represented by decimal strings separated by commas."
    )
    extra_input = (
        "Please continue the following sequence without producing any additional text. "
        "Do not say anything like 'the next terms in the sequence are', just return the numbers. "
        "Sequence:\n"
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": extra_input + input_str + settings.time_sep},
    ]

    for attempt in range(10):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                n=num_samples,
            )
            return [choice.message.content.strip() for choice in response.choices]
        except RateLimitError as e:
            msg     = str(e)
            wait_ms = re.search(r"try again in (\d+)ms", msg)
            wait_s  = re.search(r"try again in ([\d.]+)s",  msg)
            if wait_ms:
                wait = int(wait_ms.group(1)) / 1000.0 + 2.0
            elif wait_s:
                wait = float(wait_s.group(1)) + 2.0
            else:
                wait = 65.0
            print(f"\n  [Rate limit] attempt {attempt+1}: sleeping {wait:.1f}s ...")
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            wait = 10.0 * (attempt + 1)
            print(f"\n  [Timeout/ConnErr] attempt {attempt+1}: "
                  f"{type(e).__name__}, sleeping {wait:.0f}s ...")
            time.sleep(wait)

    raise RuntimeError("Mistral API failed after 10 retries.")


def _mistral_nll(input_arr, target_arr, settings, transform,
                 count_seps: bool = True, temp: float = 1.0) -> float:
    return np.nan


def _mistral_tokenize(input_str: str) -> list:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return enc.encode(input_str)


def _gpt4_nll(
    input_arr,
    target_arr,
    settings,
    transform,
    count_seps: bool = True,
    temp: float = 1.0,
) -> float:
    """
    Approximate NLL/D for GPT-4-Turbo via logprobs on the target sequence.
    Returns NaN — logprobs are not reliably available for chat models.
    """
    return np.nan


def _gpt4_tokenize(input_str: str) -> list:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    return enc.encode(input_str)


# ── Public registries used by llmtime.py ─────────────────────────────────────

completion_fns = {
    "gpt-4-turbo":       _gpt4_completion,
    "mistral-small":     _mistral_completion,
}

nll_fns = {
    "gpt-4-turbo":       _gpt4_nll,
    "mistral-small":     _mistral_nll,
}

tokenization_fns = {
    "gpt-4-turbo":       _gpt4_tokenize,
    "mistral-small":     _mistral_tokenize,
}

context_lengths = {
    "gpt-4-turbo":       128000,
    "mistral-small":     32000,
}

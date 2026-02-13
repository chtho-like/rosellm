#!/bin/bash

set -euo pipefail

BASE_URL="${ROSEINFER_BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${ROSEINFER_MODEL:-gpt2}"
MAX_TOKENS="${ROSEINFER_MAX_TOKENS:-128}"
TEMPERATURE="${ROSEINFER_TEMPERATURE:-0.7}"
TOP_P="${ROSEINFER_TOP_P:-0.9}"
SYSTEM_PROMPT="${ROSEINFER_SYSTEM_PROMPT:-}"

PY_CODE="$(cat <<'PY'
import sys
import os
from urllib.parse import urlparse


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: openai_client_chat.sh <base_url> <model> <max_tokens> "
            "<temperature> <top_p> <system_prompt>",
            file=sys.stderr,
        )
        return 2
    base_url = str(sys.argv[1])
    model = str(sys.argv[2])
    max_tokens = int(sys.argv[3])
    temperature = float(sys.argv[4])
    top_p = float(sys.argv[5])
    system_prompt = str(sys.argv[6]).strip()

    parsed = urlparse(base_url)
    host = str(parsed.hostname or "")
    if host in ("127.0.0.1", "localhost", "0.0.0.0"):
        for key in ("NO_PROXY", "no_proxy"):
            cur = str(os.environ.get(key, ""))
            parts = [p.strip() for p in cur.split(",") if p.strip()]
            for need in ("127.0.0.1", "localhost"):
                if need not in parts:
                    parts.append(need)
            os.environ[key] = ",".join(parts)

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "missing dependency: openai\n"
            "install it in your env, for example:\n"
            "  pip install openai",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(
        base_url=base_url,
        api_key="dummy",
    )

    print(f"[client] base_url: {base_url}")
    print(f"[client] model: {model}")
    print(f"[client] max_tokens: {max_tokens}")
    print(f"[client] temperature: {temperature}")
    print(f"[client] top_p: {top_p}")
    if system_prompt:
        print("[client] system prompt: enabled")
    print("[client] commands: /clear /exit /quit")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    while True:
        try:
            user_text = input("You> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            return 0
        if user_text == "/clear":
            messages = []
            print("[client] history cleared")
            continue

        messages.append({"role": "user", "content": user_text})
        print("Assistant> ", end="", flush=True)
        assistant_parts = []
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for event in stream:
            try:
                delta = event.choices[0].delta
                piece = getattr(delta, "content", None)
            except Exception:
                piece = None
            if not piece:
                continue
            assistant_parts.append(piece)
            print(piece, end="", flush=True)
        print()
        messages.append(
            {"role": "assistant", "content": "".join(assistant_parts)}
        )


if __name__ == "__main__":
    raise SystemExit(main())
PY
)"

python -c "$PY_CODE" \
  "$BASE_URL" \
  "$MODEL" \
  "$MAX_TOKENS" \
  "$TEMPERATURE" \
  "$TOP_P" \
  "$SYSTEM_PROMPT"

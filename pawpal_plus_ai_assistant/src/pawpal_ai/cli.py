from __future__ import annotations

import argparse
from pathlib import Path
import json
import os

from dotenv import load_dotenv

from .agent import PawPalAgent
from .gemini_client import DEFAULT_GEMINI_MODELS, GeminiClient
from .models import Owner
from .ollama_client import OllamaClient, OllamaConnectionError

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
DATA_FILE = BASE_DIR / "data" / "sample_owner.json"
KB_FILE = BASE_DIR / "data" / "pet_care_kb.json"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def load_owner() -> Owner:
    return Owner.load_from_json(str(DATA_FILE))


def save_owner(owner: Owner) -> None:
    owner.save_to_json(str(DATA_FILE))


def choose_retrieval_backend() -> str:
    print("Retrieval backends:")
    print("  1. TF-IDF local")
    print("  2. Gemini embeddings")
    raw = input("Select retrieval backend (default 1): ").strip() or "1"
    if raw == "2" and GeminiClient.is_configured():
        return "gemini"
    if raw == "2":
        print("Gemini key not configured. Falling back to TF-IDF.")
    return "tfidf"


def choose_llm_backend(ollama_base_url: str) -> tuple[str, str | None, str]:
    gemini_default = DEFAULT_GEMINI_MODELS[0].name
    print("Answer backends:")
    print("  1. Deterministic local agent")
    print("  2. Gemini API model")
    print("  3. Ollama local model")
    raw = input("Select answer backend (default 1): ").strip() or "1"
    if raw == "2":
        if not GeminiClient.is_configured():
            print("Gemini key not configured. Falling back to deterministic mode.")
            return "deterministic", None, gemini_default
        custom = input(f"Gemini model name (default {gemini_default}): ").strip() or gemini_default
        return "gemini", None, custom
    if raw == "3":
        client = OllamaClient(ollama_base_url)
        try:
            models = client.list_models()
        except OllamaConnectionError as exc:
            print(f"Ollama unavailable: {exc}")
            return "deterministic", None, gemini_default

        if models:
            print("Installed local Ollama models:")
            for index, model in enumerate(models, start=1):
                print(f"  {index}. {model.label()}")
        desired = input("Model name to use or pull (e.g. llama3.2): ").strip()
        if desired and desired not in [m.name for m in models]:
            pull = input(f"Model '{desired}' is not installed. Pull it now? [y/N]: ").strip().lower()
            if pull == "y":
                status = client.pull_model(desired)
                print(f"Pull result: {status}")
        if desired:
            return "ollama", desired, gemini_default
        if models:
            return "ollama", models[0].name, gemini_default
        print("No local Ollama model selected. Falling back to deterministic mode.")
    return "deterministic", None, gemini_default


def main() -> None:
    parser = argparse.ArgumentParser(description="PawPal+ AI Assistant CLI")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_URL, help="Base URL for local Ollama")
    args = parser.parse_args()

    owner = load_owner()
    retrieval_backend = choose_retrieval_backend()
    llm_backend, ollama_model, gemini_model = choose_llm_backend(args.ollama_base_url)
    gemini_client = GeminiClient() if GeminiClient.is_configured() and (llm_backend == "gemini" or retrieval_backend == "gemini") else None

    agent = PawPalAgent(
        str(KB_FILE),
        retrieval_backend=retrieval_backend,
        llm_backend=llm_backend,
        ollama_model=ollama_model,
        ollama_base_url=args.ollama_base_url,
        gemini_model=gemini_model,
        gemini_client=gemini_client,
    )

    print("PawPal+ AI Assistant CLI")
    print("========================")
    print(f"Retrieval: {retrieval_backend}")
    if llm_backend == "ollama" and ollama_model:
        print(f"Answer backend: Ollama ({ollama_model})")
    elif llm_backend == "gemini":
        print(f"Answer backend: Gemini ({gemini_model})")
    else:
        print("Answer backend: Deterministic local agent")
    print("Type a natural-language request. Examples:")
    print("- Schedule a 30 minute morning walk for Mochi tomorrow at 7am")
    print("- Check conflicts")
    print("- What should I focus on today?")
    print("- Find a 20 minute open slot tomorrow")
    print("- quit")
    print()

    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        response = agent.run(query, owner)
        print("\nAssistant:")
        print(response.answer)
        print("\nStructured JSON:")
        print(json.dumps(response.to_dict(), indent=2))

        if response.proposed_tasks:
            choice = input("\nApply proposed tasks? [y/N]: ").strip().lower()
            if choice == "y":
                outcome = agent.apply_proposals(response, owner)
                save_owner(owner)
                print(json.dumps(outcome, indent=2))
        print("\n" + "-" * 72 + "\n")

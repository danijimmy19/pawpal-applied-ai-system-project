PawPal+ AI Assistant is a rebuilt final-project version of the original **PawPal+ pet care scheduler**. The base project already managed pets, tasks, sorting, conflict checks, recurring tasks, JSON persistence, a CLI demo, and a Streamlit interface. This extension turns that scheduler into a grounded AI system with retrieval, an observable agent workflow, guardrails, and an evaluation harness.

## Why this project matters

Pet owners often need two things at once:

1. reliable scheduling for daily care tasks
2. quick, grounded help when they ask natural-language questions like “What should I focus on today?” or “Schedule a 30 minute walk for Mochi tomorrow at 7am.”

This project combines both in one system.

## Final-project features

### 1) RAG system
The assistant retrieves from two sources before answering:

- a curated local pet-care knowledge base
- the live pet/task schedule from the current owner profile

The retriever can use **TF-IDF + cosine similarity** locally or **Gemini embeddings** when a Gemini API key is configured, so users can choose a fully local path or a semantic cloud-retrieval path.

### 1b) Flexible model selection
Users can choose among three answer modes in the Streamlit sidebar or CLI startup flow:

- **Deterministic local agent** for fully reproducible grading-friendly behavior
- **Gemini** for API-backed structured extraction and answer synthesis
- **Ollama** for fully local model-backed extraction and answer synthesis

If the user selects an Ollama model that is not installed yet, the app can trigger an Ollama pull from the UI with one click.

### 2) Agentic workflow
The assistant uses a deterministic **ReAct-style workflow** with an observable trace:

- classify the request
- retrieve context
- choose a tool
- inspect tool output
- self-critique and assign confidence

Supported actions:
- add a task proposal
- review the schedule
- check conflicts
- find the next available slot
- answer grounded care questions

### 3) Guardrails
The system includes:
- prompt-injection detection
- medical-emergency redirect
- structured task validation
- iteration caps
- structured JSON output
- self-critique with confidence scoring

### 4) Evaluation harness
A separate evaluation script runs **12 scenarios** across the live system and prints:
- pass/fail by case
- average confidence
- a final summary line

## Architecture overview

The system is organized like this:

1. **UI / CLI**
   - Streamlit app and CLI both call the same backend.

2. **Agent**
   - The `PawPalAgent` class handles intent detection, retrieval, tool choice, self-critique, and structured output.

3. **Retriever**
   - `RAGRetriever` supports a TF-IDF backend and a Gemini-embeddings backend over the knowledge base plus live schedule-derived documents.

4. **Model clients**
   - `gemini_client.py` handles Gemini embeddings and structured JSON generation.
   - `ollama_client.py` lists local models, pulls missing models, and calls the local chat endpoint.

5. **Schedule tools**
   - `Scheduler` provides sorting, filtering, conflict detection, completion logic, and slot finding.

6. **Guardrails**
   - Input safety checks run before tool execution.
   - Task proposals are validated before any schedule update.

7. **Evaluation**
   - `eval/run_evaluation.py` exercises the full system on multiple scenarios.

See the PNG architecture diagram in `assets/system_architecture.png`.

## Project structure

```text
pawpal_plus_ai_assistant/
├── assets/
│   └── system_architecture.png
├── data/
│   ├── pet_care_kb.json
│   └── sample_owner.json
├── eval/
│   ├── eval_cases.json
│   └── run_evaluation.py
├── src/
│   └── pawpal_ai/
│       ├── __init__.py
│       ├── agent.py
│       ├── app.py
│       ├── cli.py
│       ├── guardrails.py
│       ├── models.py
│       ├── ollama_client.py
│       ├── retrieval.py
│       └── scheduler.py
├── tests/
│   ├── test_agent.py
│   ├── test_ollama_mode.py
│   └── test_scheduler.py
├── main.py
├── run_streamlit.py
├── .env.example
├── README.md
├── reflection.md
└── requirements.txt
```

## Setup instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the CLI

```bash
python main.py
```

### 4. Start Ollama for fully local model-backed inference (optional)

Install Ollama, start the local server, and pull at least one model such as `llama3.2` or `gemma3`. Then run:

```bash
ollama serve
ollama pull llama3.2
```

### 5. Configure optional Gemini access

Copy `.env.example` to `.env` and add your Gemini key if you want Gemini retrieval or Gemini answer synthesis.

### 6. Run the Streamlit app

```bash
streamlit run run_streamlit.py
```

In the sidebar you can mix and match:
- **Retrieval:** TF-IDF local or Gemini embeddings
- **Answers:** Deterministic, Gemini, or Ollama

If you type an Ollama model name that is not installed yet, click **Download / pull Ollama model** and the app will install it before use.

### 7. Run the tests

```bash
PYTHONPATH=src python -m pytest
```

### 8. Run the evaluation harness

```bash
PYTHONPATH=src python eval/run_evaluation.py
```

## Sample interactions

### Example 1 — natural-language task planning
Input:
```text
Schedule a 30 minute morning walk for Mochi tomorrow at 7am
```

Behavior:
- intent = `add_task`
- retrieves exercise/schedule context
- creates a structured task proposal
- waits for approval before modifying the schedule

### Example 2 — schedule review
Input:
```text
What should I focus on today?
```

Behavior:
- intent = `schedule_review`
- summarizes pending tasks, high-priority items, and conflicts

### Example 3 — medical safety redirect
Input:
```text
My dog collapsed and has trouble breathing
```

Behavior:
- triggers the medical-emergency guardrail
- blocks tool execution
- redirects the user to immediate veterinary care

## Design decisions and tradeoffs

### Why keep the deterministic backend even after adding Ollama?
The deterministic backend makes the demo reproducible for grading and serves as a safe fallback when Ollama is not running. The optional Ollama integration gives users a fully local model-backed mode without making the project brittle.

### Why show a trace instead of hidden reasoning?
The app exposes an **action trace** with:
- step number
- thought summary
- action
- observation

That gives the project observable agent behavior without depending on raw hidden chain-of-thought.

### Why keep task application human-approved?
The assistant proposes schedule changes first and applies them only after approval. This is safer than auto-writing tasks from ambiguous natural language.

### Why use local JSON data?
It keeps the project easy to run and inspect. The tradeoff is that persistence is local rather than multi-user or cloud-backed.

## Testing summary

This rebuild includes:
- unit tests for scheduler behavior
- unit tests for the AI agent and guardrails
- a 12-case evaluation harness

Typical checks:
- task proposal extraction
- emergency redirect
- prompt-injection blocking
- schedule review
- slot finding
- conflict checking

## Reflection on AI collaboration
[Need to add]

## Limitations and ethics

- This system is **not** a veterinary diagnostic tool.
- It gives general care guidance and redirects emergencies, but it cannot replace a licensed veterinarian.
- The knowledge base is intentionally narrow and conservative.
- A user could still ask ambiguous questions, so the system keeps confidence scoring and approval gates rather than pretending certainty.


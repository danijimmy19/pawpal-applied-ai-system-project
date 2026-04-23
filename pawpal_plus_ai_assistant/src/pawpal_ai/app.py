from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

load_dotenv(BASE_DIR / ".env")

from pawpal_ai.agent import PawPalAgent
from pawpal_ai.gemini_client import DEFAULT_GEMINI_MODELS, GeminiClient, GeminiConnectionError
from pawpal_ai.models import Owner, Pet, Task
from pawpal_ai.ollama_client import OllamaClient, OllamaConnectionError
from pawpal_ai.scheduler import Scheduler

DATA_FILE = BASE_DIR / "data" / "sample_owner.json"
KB_FILE = BASE_DIR / "data" / "pet_care_kb.json"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

st.set_page_config(page_title="PawPal+ AI Assistant", page_icon="🐾", layout="wide")
st.title("🐾 PawPal+ AI Assistant")
st.caption("TF-IDF or Gemini retrieval + deterministic, Gemini, or Ollama answers with guardrails and scheduler tools")


def load_owner_from_disk() -> Owner:
    return Owner.load_from_json(str(DATA_FILE))


def save_owner(owner: Owner) -> None:
    owner.save_to_json(str(DATA_FILE))


def fetch_ollama_models(base_url: str) -> tuple[list[str], str | None]:
    client = OllamaClient(base_url)
    try:
        models = client.list_models()
        return [model.name for model in models], None
    except OllamaConnectionError as exc:
        return [], str(exc)


if "owner" not in st.session_state:
    st.session_state.owner = load_owner_from_disk()
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "ollama_refresh_nonce" not in st.session_state:
    st.session_state.ollama_refresh_nonce = 0
if "ollama_selected_model" not in st.session_state:
    st.session_state.ollama_selected_model = ""

owner: Owner = st.session_state.owner
scheduler = Scheduler(owner)

gemini_available = GeminiClient.is_configured()

st.sidebar.header("Inference settings")
retrieval_label = st.sidebar.radio(
    "Retrieval backend",
    ["TF-IDF local", "Gemini embeddings"],
    index=0,
    help="TF-IDF is fully local. Gemini embeddings use your Gemini API key for semantic retrieval.",
)
retrieval_backend = "gemini" if retrieval_label == "Gemini embeddings" else "tfidf"
if retrieval_backend == "gemini" and not gemini_available:
    st.sidebar.warning("Gemini embeddings selected, but no GEMINI_API_KEY / GOOGLE_API_KEY was found. Falling back to TF-IDF.")
    retrieval_backend = "tfidf"

backend_label = st.sidebar.radio(
    "Answer backend",
    ["Deterministic local agent", "Gemini API model", "Ollama local model"],
    index=0,
    help="Pick how the agent extracts tasks and polishes grounded answers.",
)

llm_backend = "deterministic"
ollama_base_url = DEFAULT_OLLAMA_URL
selected_ollama_model: str | None = None
gemini_model = DEFAULT_GEMINI_MODELS[0].name

if backend_label == "Gemini API model":
    llm_backend = "gemini"
    if not gemini_available:
        st.sidebar.error("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env to use Gemini.")
        llm_backend = "deterministic"
    else:
        gemini_default_names = [item.name for item in DEFAULT_GEMINI_MODELS]
        gemini_model = st.sidebar.selectbox("Gemini model", gemini_default_names, index=0)
        custom_gemini_model = st.sidebar.text_input("Or enter a custom Gemini model", value="")
        if custom_gemini_model.strip():
            gemini_model = custom_gemini_model.strip()
        st.sidebar.success(f"Using Gemini model: {gemini_model}")
elif backend_label == "Ollama local model":
    llm_backend = "ollama"
    ollama_base_url = st.sidebar.text_input("Ollama base URL", value=DEFAULT_OLLAMA_URL)
    if st.sidebar.button("Refresh local models"):
        st.session_state.ollama_refresh_nonce += 1

    available_models, ollama_error = fetch_ollama_models(ollama_base_url)
    model_to_pull = st.sidebar.text_input(
        "Ollama model to use or download",
        value=st.session_state.ollama_selected_model or (available_models[0] if available_models else "llama3.2"),
        help="Examples: llama3.2, gemma3, qwen3",
    ).strip()
    if model_to_pull:
        st.session_state.ollama_selected_model = model_to_pull

    if available_models:
        selected_from_list = st.sidebar.selectbox(
            "Installed Ollama model",
            available_models,
            index=available_models.index(model_to_pull) if model_to_pull in available_models else 0,
        )
        selected_ollama_model = selected_from_list
        st.session_state.ollama_selected_model = selected_from_list
    elif ollama_error:
        st.sidebar.warning(ollama_error)

    if st.sidebar.button("Download / pull Ollama model"):
        if not model_to_pull:
            st.sidebar.error("Enter a model name first.")
        else:
            try:
                with st.spinner(f"Pulling {model_to_pull} from Ollama..."):
                    status = OllamaClient(ollama_base_url).pull_model(model_to_pull)
                st.session_state.ollama_selected_model = model_to_pull
                selected_ollama_model = model_to_pull
                st.sidebar.success(f"Ollama pull result: {status}")
            except OllamaConnectionError as exc:
                st.sidebar.error(str(exc))

    if selected_ollama_model:
        st.sidebar.success(f"Using local model: {selected_ollama_model}")
    elif model_to_pull:
        st.sidebar.info("Click 'Download / pull Ollama model' to install and use that model locally.")
        llm_backend = "deterministic"
        st.sidebar.caption("Falling back to deterministic mode until a local model is installed.")

try:
    gemini_client = GeminiClient() if gemini_available and (llm_backend == "gemini" or retrieval_backend == "gemini") else None
except GeminiConnectionError as exc:
    st.sidebar.warning(str(exc))
    gemini_client = None
    if llm_backend == "gemini":
        llm_backend = "deterministic"
    if retrieval_backend == "gemini":
        retrieval_backend = "tfidf"

agent = PawPalAgent(
    str(KB_FILE),
    retrieval_backend=retrieval_backend,
    llm_backend=llm_backend,
    ollama_model=selected_ollama_model,
    ollama_base_url=ollama_base_url,
    gemini_model=gemini_model,
    gemini_client=gemini_client,
)

st.sidebar.markdown("---")
st.sidebar.write(f"Retrieval: **{retrieval_backend}**")
st.sidebar.write(f"Answers: **{llm_backend}**")
if llm_backend == "ollama" and selected_ollama_model:
    st.sidebar.write(f"Ollama model: **{selected_ollama_model}**")
if llm_backend == "gemini":
    st.sidebar.write(f"Gemini model: **{gemini_model}**")


tab_scheduler, tab_ai, tab_docs = st.tabs(["Scheduler", "AI Assistant", "About"])

with tab_scheduler:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Owner")
        with st.form("owner_form"):
            owner_name = st.text_input("Owner name", value=owner.name)
            owner_email = st.text_input("Owner email", value=owner.email)
            owner_submit = st.form_submit_button("Update owner")
        if owner_submit:
            owner.name = owner_name
            owner.email = owner_email
            save_owner(owner)
            st.success("Owner updated.")

        st.subheader("Add a pet")
        with st.form("pet_form", clear_on_submit=True):
            pet_name = st.text_input("Pet name")
            species = st.selectbox("Species", ["Dog", "Cat", "Other"])
            age = st.number_input("Age", min_value=0, max_value=40, value=1)
            notes = st.text_input("Notes")
            pet_submit = st.form_submit_button("Add pet")
        if pet_submit:
            if not pet_name.strip():
                st.error("Pet name is required.")
            elif owner.get_pet(pet_name.strip()):
                st.warning("That pet already exists.")
            else:
                owner.add_pet(Pet(name=pet_name.strip(), species=species, age=int(age), notes=notes.strip()))
                save_owner(owner)
                st.success(f"Added {pet_name.strip()}.")

        st.subheader("Add a task")
        if owner.pets:
            with st.form("task_form", clear_on_submit=True):
                chosen_pet = st.selectbox("Assign to pet", [pet.name for pet in owner.pets], key="manual_task_pet")
                description = st.text_input("Task description")
                due_date_value = st.date_input("Due date", value=date.today())
                due_time_value = st.time_input(
                    "Due time", value=datetime.now().replace(second=0, microsecond=0).time()
                )
                duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=30)
                priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)
                frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])
                task_type = st.selectbox(
                    "Task type", ["feeding", "exercise", "medical", "grooming", "appointment", "general"]
                )
                task_submit = st.form_submit_button("Add task")
            if task_submit:
                pet = owner.get_pet(chosen_pet)
                if pet is None:
                    st.error("Selected pet was not found.")
                elif not description.strip():
                    st.error("Task description is required.")
                else:
                    pet.add_task(
                        Task(
                            description=description.strip(),
                            due_date=due_date_value,
                            due_time=due_time_value,
                            frequency=frequency,
                            priority=priority,
                            duration_minutes=int(duration),
                            task_type=task_type,
                        )
                    )
                    save_owner(owner)
                    st.success(f"Added '{description.strip()}' for {chosen_pet}.")
        else:
            st.info("Add a pet before adding tasks.")

    with col_right:
        st.subheader("Schedule")
        sort_mode = st.radio(
            "Sort mode",
            ["time", "priority"],
            horizontal=True,
            format_func=lambda value: "By time" if value == "time" else "By priority",
            key="sort_mode",
        )
        show_completed = st.toggle("Show completed tasks", value=True, key="show_completed_toggle")
        rows = scheduler.agenda_table(sort_mode=sort_mode, include_completed=show_completed)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No tasks yet.")

        conflicts = scheduler.detect_conflicts()
        if conflicts:
            for warning in conflicts:
                st.warning(warning)
        else:
            st.success("No conflicts detected.")

        st.subheader("Mark a task complete")
        pending_options = [
            f"{pet.name} — {task.description} ({task.due_date.isoformat()} {task.due_time.strftime('%H:%M')})"
            for pet, task in scheduler.get_all_tasks(include_completed=False)
        ]
        if pending_options:
            chosen_label = st.selectbox("Pending tasks", pending_options, key="pending_select")
            if st.button("Complete selected task", key="complete_task_button"):
                pet_name, remainder = chosen_label.split(" — ", maxsplit=1)
                description = remainder.split(" (", maxsplit=1)[0]
                updated = scheduler.mark_task_complete(pet_name, description)
                if updated:
                    save_owner(owner)
                    st.success(f"Completed '{description}' for {pet_name}.")
                else:
                    st.error("Could not complete that task.")
        else:
            st.info("No pending tasks.")

        st.subheader("Find next available slot")
        with st.form("slot_form"):
            slot_date = st.date_input("Date", value=date.today(), key="slot_date")
            slot_duration = st.number_input("Needed minutes", min_value=1, max_value=240, value=20, key="slot_duration")
            slot_submit = st.form_submit_button("Find slot")
        if slot_submit:
            slot = scheduler.next_available_slot(slot_date, int(slot_duration))
            if slot:
                st.success(f"Next available slot: {slot.strftime('%H:%M')}")
            else:
                st.warning("No open slot found in the default 08:00–20:00 window.")

with tab_ai:
    st.subheader("Ask the AI assistant")
    st.write(
        "Examples: “Schedule a 30 minute morning walk for Mochi tomorrow at 7am”, "
        "“Check conflicts”, “Find a 20 minute open slot tomorrow”, "
        "“Simba is vomiting and not eating. What should I do?”"
    )
    if llm_backend == "ollama" and selected_ollama_model:
        st.info(f"Answer synthesis and task extraction use local Ollama model `{selected_ollama_model}`.")
    elif llm_backend == "gemini":
        st.info(f"Answer synthesis and task extraction use Gemini model `{gemini_model}`.")
    else:
        st.caption("Using deterministic local agent mode. No external or model-server calls are required for answers.")

    with st.form("ai_form"):
        query = st.text_area("Your request", height=120, key="ai_query")
        submitted = st.form_submit_button("Analyze request")

    if submitted:
        try:
            result = agent.run(query, owner)
            st.session_state.last_analysis = result
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.last_analysis
    if result is not None:
        st.markdown("### Answer")
        st.write(result.answer)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intent", result.intent)
        c2.metric("Confidence", f"{result.confidence:.2f}")
        c3.metric("Guardrail", result.guardrail_status)
        c4.metric("Backend", result.tool_outputs.get("llm_backend", llm_backend))

        st.markdown("### Retrieved context")
        for chunk in result.retrieved_context:
            st.write(f"**{chunk['source']} — {chunk['title']}** ({chunk['score']:.2f})")
            st.caption(chunk["text"])

        st.markdown("### Observable trace")
        for step in result.trace:
            with st.expander(f"Step {step['step']}: {step['action']}"):
                st.write(f"**Thought:** {step['thought']}")
                st.write(f"**Observation:** {step['observation']}")

        if result.proposed_tasks:
            st.markdown("### Proposed tasks")
            st.dataframe(result.proposed_tasks, use_container_width=True, hide_index=True)
            if st.button("Apply proposed tasks"):
                outcome = agent.apply_proposals(result, owner)
                save_owner(owner)
                st.success(f"Applied {outcome['applied_count']} tasks.")
                if outcome["skipped"]:
                    st.warning(f"Skipped {len(outcome['skipped'])} task(s).")
                st.json(outcome)

        st.markdown("### Self-critique")
        st.json(result.self_critique)
        st.markdown("### Full structured output")
        st.json(result.to_dict())

with tab_docs:
    st.subheader("About this system")
    st.markdown(
        """
- **Retrieval options:** local TF-IDF or Gemini embeddings
- **Answer options:** deterministic local rules, Gemini, or Ollama
- **Agent workflow:** classify → retrieve → tool use → synthesize → self-critique
- **Guardrails:** prompt-injection checks, medical redirect, structured payload validation, step cap
- **Scheduler tools:** add tasks, detect conflicts, review schedule, find next slot
        """
    )

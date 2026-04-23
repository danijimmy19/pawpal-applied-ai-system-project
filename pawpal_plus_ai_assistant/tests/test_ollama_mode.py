from datetime import date, time
from pathlib import Path

from pawpal_ai.agent import PawPalAgent
from pawpal_ai.models import Owner, Pet, Task


class FakeOllamaClient:
    def chat_json(self, *, model, messages, schema, options=None, keep_alive="5m", think=False):
        del model, messages, options, keep_alive, think
        properties = schema.get("properties", {})
        if "proposals" in properties:
            tomorrow = date.today().replace(day=date.today().day).isoformat()
            # use tomorrow-like deterministic shape without depending on calendar math in the test
            return {
                "proposals": [
                    {
                        "pet_name": "Mochi",
                        "description": "Morning Walk",
                        "due_date": tomorrow,
                        "due_time": "07:00",
                        "frequency": "daily",
                        "priority": "medium",
                        "duration_minutes": 30,
                        "task_type": "exercise",
                    }
                ]
            }
        return {
            "answer": "Local Ollama model synthesized a grounded answer.",
            "warnings": ["Used local model synthesis."],
        }


class FailingOllamaClient:
    def chat_json(self, **kwargs):
        raise RuntimeError("Ollama is unavailable")


def build_owner() -> Owner:
    owner = Owner(name="Belem")
    mochi = Pet(name="Mochi", species="Dog", age=3, notes="Needs daily exercise")
    owner.add_pet(mochi)
    mochi.add_task(Task("Breakfast", date.today(), time(8, 0), priority="high", task_type="feeding"))
    return owner


def test_agent_uses_ollama_for_local_model_flow() -> None:
    kb_path = Path(__file__).resolve().parents[1] / "data" / "pet_care_kb.json"
    agent = PawPalAgent(
        str(kb_path),
        llm_backend="ollama",
        ollama_model="llama3.2",
        ollama_client=FakeOllamaClient(),
    )
    response = agent.run("Schedule a 30 minute morning walk for Mochi tomorrow at 7am", build_owner())

    assert response.proposed_tasks
    assert response.tool_outputs["model_used"] == "llama3.2"
    assert "Local Ollama model synthesized" in response.answer
    assert any(step["action"] == "ollama_extract_task_proposals" for step in response.trace)
    assert any(step["action"] == "ollama_answer_synthesis" for step in response.trace)


def test_agent_falls_back_when_ollama_fails() -> None:
    kb_path = Path(__file__).resolve().parents[1] / "data" / "pet_care_kb.json"
    agent = PawPalAgent(
        str(kb_path),
        llm_backend="ollama",
        ollama_model="llama3.2",
        ollama_client=FailingOllamaClient(),
    )
    response = agent.run("What should I focus on today?", build_owner())

    assert response.intent == "schedule_review"
    assert "Pending tasks" in response.answer
    assert any("Ollama answer synthesis failed" in warning for warning in response.self_critique["warnings"])

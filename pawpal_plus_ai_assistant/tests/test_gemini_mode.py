from datetime import date, time
from pathlib import Path

from pawpal_ai.agent import PawPalAgent
from pawpal_ai.models import Owner, Pet, Task


class FakeGeminiClient:
    def embed_texts(self, *, texts, task_type, titles=None, output_dimensionality=768, model="gemini-embedding-001"):
        del titles, output_dimensionality, model
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "medication" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "walk" in lowered or "exercise" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        if task_type == "RETRIEVAL_QUERY":
            return [vectors[0]]
        return vectors

    def generate_json(self, *, model, prompt, schema):
        del model, prompt
        if "proposals" in schema.get("properties", {}):
            tomorrow = date.today().isoformat()
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
            "answer": "Gemini synthesized a grounded answer.",
            "warnings": ["Used Gemini model synthesis."],
        }


def build_owner() -> Owner:
    owner = Owner(name="Belem")
    mochi = Pet(name="Mochi", species="Dog", age=3, notes="Needs daily exercise")
    simba = Pet(name="Simba", species="Cat", age=5, notes="Takes medication")
    owner.add_pet(mochi)
    owner.add_pet(simba)
    mochi.add_task(Task("Breakfast", date.today(), time(8, 0), priority="high", task_type="feeding"))
    simba.add_task(Task("Medication", date.today(), time(19, 0), priority="high", task_type="medical"))
    return owner


def test_agent_uses_gemini_for_retrieval_and_answering() -> None:
    kb_path = Path(__file__).resolve().parents[1] / "data" / "pet_care_kb.json"
    agent = PawPalAgent(
        str(kb_path),
        retrieval_backend="gemini",
        llm_backend="gemini",
        gemini_model="gemini-2.5-flash",
        gemini_client=FakeGeminiClient(),
    )
    response = agent.run("Schedule a 30 minute morning walk for Mochi tomorrow at 7am", build_owner())

    assert response.proposed_tasks
    assert response.tool_outputs["model_used"] == "gemini-2.5-flash"
    assert response.tool_outputs["retrieval_backend"] == "gemini"
    assert "Gemini synthesized" in response.answer
    assert any(step["action"] == "gemini_extract_task_proposals" for step in response.trace)
    assert any(step["action"] == "gemini_answer_synthesis" for step in response.trace)

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import json
import re
from typing import Any, Dict, List, Optional

from .gemini_client import GeminiClient, GeminiConnectionError
from .guardrails import detect_prompt_injection, medical_safety_check, validate_task_payload
from .models import Owner, Task
from .ollama_client import OllamaClient, OllamaConnectionError
from .retrieval import RAGRetriever, RetrievedChunk
from .scheduler import Scheduler


@dataclass
class AgentTraceStep:
    step: int
    thought: str
    action: str
    observation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskProposal:
    pet_name: str
    description: str
    due_date: str
    due_time: str
    frequency: str = "once"
    priority: str = "medium"
    duration_minutes: int = 30
    task_type: str = "general"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_task(self) -> Task:
        return Task(
            description=self.description,
            due_date=date.fromisoformat(self.due_date),
            due_time=datetime.strptime(self.due_time, "%H:%M").time(),
            frequency=self.frequency,
            priority=self.priority,
            duration_minutes=self.duration_minutes,
            task_type=self.task_type,
        )


@dataclass
class AgentResponse:
    intent: str
    answer: str
    confidence: float
    guardrail_status: str
    retrieved_context: List[dict]
    trace: List[dict]
    proposed_tasks: List[dict] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    self_critique: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PawPalAgent:
    """Scheduling agent with configurable retrieval and answer backends."""

    def __init__(
        self,
        kb_path: str,
        max_steps: int = 6,
        *,
        retrieval_backend: str = "tfidf",
        llm_backend: str = "deterministic",
        ollama_model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_client: Optional[OllamaClient] = None,
        gemini_model: str = "gemini-2.5-flash",
        gemini_embedding_model: str = "gemini-embedding-001",
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.retriever = RAGRetriever(
            kb_path,
            backend=retrieval_backend,
            gemini_client=gemini_client,
            gemini_embedding_model=gemini_embedding_model,
        )
        self.max_steps = max_steps
        self.llm_backend = llm_backend
        self.retrieval_backend = retrieval_backend
        self.ollama_model = ollama_model
        self.ollama_client = ollama_client or OllamaClient(ollama_base_url)
        self.gemini_model = gemini_model
        self.gemini_embedding_model = gemini_embedding_model

    def run(self, query: str, owner: Owner) -> AgentResponse:
        trace: list[AgentTraceStep] = []
        scheduler = Scheduler(owner)
        tool_outputs: dict[str, Any] = {
            "llm_backend": self.llm_backend,
            "retrieval_backend": self.retrieval_backend,
        }

        injection = detect_prompt_injection(query)
        if not injection.allowed:
            trace.append(self._step(1, "Check for prompt injection.", "guardrail_scan", injection.status))
            return self._blocked_response(
                intent="blocked",
                answer=injection.user_message,
                guardrail_status=injection.status,
                trace=trace,
                tool_outputs=tool_outputs,
            )

        medical = medical_safety_check(query)
        trace.append(
            self._step(
                1,
                "Check the request for safety risks.",
                "medical_safety_check",
                medical.status if medical.status != "ok" else "no medical red flags",
            )
        )
        if not medical.allowed:
            return self._blocked_response(
                intent="medical_redirect",
                answer=medical.user_message,
                guardrail_status=medical.status,
                trace=trace,
                tool_outputs=tool_outputs,
            )

        intent = self._classify_intent(query)
        trace.append(self._step(2, "Decide what kind of request this is.", "classify_intent", intent))

        retrieved: list[RetrievedChunk]
        retrieval_warning = None
        try:
            retrieved = self.retriever.retrieve(query, owner, top_k=5)
        except GeminiConnectionError as exc:
            self.retriever.set_backend("tfidf")
            self.retrieval_backend = "tfidf"
            tool_outputs["retrieval_backend"] = "tfidf"
            retrieved = self.retriever.retrieve(query, owner, top_k=5)
            retrieval_warning = f"Gemini retrieval unavailable; fell back to TF-IDF: {exc}"
        trace.append(
            self._step(
                3,
                "Retrieve relevant pet-care and schedule context.",
                "retrieve_context",
                f"Retrieved {len(retrieved)} chunks via {tool_outputs['retrieval_backend']}",
            )
        )

        proposed_tasks: list[TaskProposal] = []
        answer = ""
        self_critique: dict[str, Any] = {"checks": [], "warnings": []}
        if retrieval_warning:
            self_critique["warnings"].append(retrieval_warning)
        step_no = 4

        if intent == "add_task":
            proposals, used_model, warnings = self._extract_task_proposals_with_optional_model(query, owner, retrieved)
            proposed_tasks = proposals
            self_critique["warnings"].extend(warnings)
            action_name = (
                "ollama_extract_task_proposals" if used_model == "ollama"
                else "gemini_extract_task_proposals" if used_model == "gemini"
                else "extract_task_proposals"
            )
            trace.append(
                self._step(
                    step_no,
                    "Parse structured task data from the request.",
                    action_name,
                    f"Parsed {len(proposed_tasks)} proposals.",
                )
            )
            step_no += 1

            validations = []
            valid_proposals = []
            for proposal in proposed_tasks:
                validation = validate_task_payload(proposal.to_dict())
                if not owner.get_pet(proposal.pet_name):
                    validation.errors.append(f"pet '{proposal.pet_name}' not found")
                    validation.valid = False
                validations.append(validation.to_dict())
                if validation.valid:
                    valid_proposals.append(proposal)
                if validation.errors:
                    self_critique["warnings"].extend(validation.errors)

            tool_outputs["proposal_validation"] = validations
            if valid_proposals:
                answer = self._summarize_task_proposals(valid_proposals)
                self_critique["checks"].append("Task proposals are structurally valid and mapped to known pets.")
            else:
                answer = (
                    "I could not create a valid task proposal from that request. "
                    "Try including the pet name, task, date, time, and duration."
                )
        elif intent == "find_slot":
            slot_date = self._parse_date(query)
            duration = self._parse_duration(query) or 30
            slot = scheduler.next_available_slot(slot_date, duration_minutes=duration)
            tool_outputs["next_available_slot"] = slot.strftime("%H:%M") if slot else None
            trace.append(
                self._step(
                    step_no,
                    "Use the scheduling tool to find an open window.",
                    "next_available_slot",
                    tool_outputs["next_available_slot"] or "no slot found",
                )
            )
            step_no += 1
            answer = (
                f"The next available {duration}-minute slot on {slot_date.isoformat()} is {slot.strftime('%H:%M')}."
                if slot
                else f"I could not find an open {duration}-minute slot on {slot_date.isoformat()}."
            )
            self_critique["checks"].append("Slot recommendation came from live schedule data.")
        elif intent == "conflict_check":
            conflicts = scheduler.detect_conflicts()
            tool_outputs["conflicts"] = conflicts
            trace.append(
                self._step(
                    step_no,
                    "Inspect the current schedule for overlaps.",
                    "detect_conflicts",
                    f"{len(conflicts)} conflicts found.",
                )
            )
            step_no += 1
            answer = "No conflicts detected." if not conflicts else "Conflicts found:\n- " + "\n- ".join(conflicts)
            self_critique["checks"].append("Conflict scan used live task windows and overlap logic.")
        elif intent == "schedule_review":
            summary = scheduler.summarize_schedule(on_date=self._parse_date(query, default_today=True))
            tool_outputs["schedule_summary"] = asdict(summary)
            trace.append(
                self._step(
                    step_no,
                    "Summarize the current schedule state.",
                    "summarize_schedule",
                    f"{summary.pending_count} pending, {len(summary.conflicts)} conflicts.",
                )
            )
            step_no += 1
            answer = self._format_schedule_review(summary)
            self_critique["checks"].append("Review grounded in current schedule state.")
        else:
            answer = self._compose_grounded_answer(query, retrieved)
            self_critique["checks"].append("Answer grounded in retrieved schedule and knowledge-base context.")
            if medical.status == "medical_caution":
                self_critique["warnings"].append(
                    "Medical caution language detected; user should contact a vet if symptoms persist."
                )

        model_used = None
        if self._should_use_model():
            try:
                answer, model_warnings, model_used = self._synthesize_answer_with_model(
                    query=query,
                    intent=intent,
                    retrieved=retrieved,
                    draft_answer=answer,
                    tool_outputs=tool_outputs,
                    self_critique=self_critique,
                )
                tool_outputs["model_used"] = model_used
                if model_warnings:
                    self_critique["warnings"].extend(model_warnings)
                trace.append(
                    self._step(
                        step_no,
                        "Use the selected model to polish the grounded answer.",
                        f"{self.llm_backend}_answer_synthesis",
                        f"generated with {model_used}",
                    )
                )
                step_no += 1
            except Exception as exc:
                trace.append(
                    self._step(
                        step_no,
                        "Try the selected model for grounded answer synthesis.",
                        f"{self.llm_backend}_answer_synthesis",
                        f"fallback to deterministic answer: {exc}",
                    )
                )
                self_critique["warnings"].append(
                    f"{self.llm_backend.title()} answer synthesis failed; returned deterministic answer."
                )
                step_no += 1

        critique_score = self._score_confidence(intent, query, retrieved, proposed_tasks, self_critique)
        self_critique["confidence_basis"] = critique_score["basis"]
        confidence = critique_score["score"]

        trace.append(
            self._step(
                min(step_no, self.max_steps),
                "Review the answer before returning it.",
                "self_critique",
                f"confidence={confidence:.2f}",
            )
        )

        if medical.status == "medical_caution" and answer:
            answer = medical.user_message + "\n\n" + answer

        return AgentResponse(
            intent=intent,
            answer=answer,
            confidence=confidence,
            guardrail_status=medical.status,
            retrieved_context=[chunk.to_dict() for chunk in retrieved],
            trace=[step.to_dict() for step in trace[: self.max_steps]],
            proposed_tasks=[proposal.to_dict() for proposal in proposed_tasks],
            tool_outputs=tool_outputs,
            self_critique=self_critique,
            blocked=False,
        )

    def apply_proposals(self, response: AgentResponse, owner: Owner) -> dict:
        scheduler = Scheduler(owner)
        applied = []
        skipped = []
        for payload in response.proposed_tasks:
            validation = validate_task_payload(payload)
            if not validation.valid:
                skipped.append({"payload": payload, "reason": validation.errors})
                continue
            proposal = TaskProposal(**payload)
            if scheduler.add_task_to_pet(proposal.pet_name, proposal.to_task()):
                applied.append(payload)
            else:
                skipped.append({"payload": payload, "reason": [f"pet '{proposal.pet_name}' not found"]})
        return {"applied_count": len(applied), "applied": applied, "skipped": skipped}

    def _should_use_model(self) -> bool:
        if self.llm_backend == "ollama":
            return bool(self.ollama_model)
        if self.llm_backend == "gemini":
            return True
        return False

    def _blocked_response(
        self,
        intent: str,
        answer: str,
        guardrail_status: str,
        trace: list[AgentTraceStep],
        tool_outputs: Optional[dict] = None,
    ) -> AgentResponse:
        return AgentResponse(
            intent=intent,
            answer=answer,
            confidence=0.99,
            guardrail_status=guardrail_status,
            retrieved_context=[],
            trace=[step.to_dict() for step in trace],
            proposed_tasks=[],
            tool_outputs=tool_outputs or {},
            self_critique={"checks": ["Guardrail triggered before tool use."], "warnings": [guardrail_status]},
            blocked=True,
        )

    def _classify_intent(self, query: str) -> str:
        lowered = query.lower()
        if any(word in lowered for word in ["add ", "schedule ", "set up ", "remind", "every day", "every week"]):
            return "add_task"
        if "slot" in lowered or "available time" in lowered or "open time" in lowered:
            return "find_slot"
        if "conflict" in lowered or "overlap" in lowered:
            return "conflict_check"
        if any(word in lowered for word in ["review", "focus on", "today", "agenda", "schedule"]):
            return "schedule_review"
        return "care_guidance"

    def _extract_task_proposals_with_optional_model(
        self,
        query: str,
        owner: Owner,
        retrieved: list[RetrievedChunk],
    ) -> tuple[list[TaskProposal], str, list[str]]:
        warnings: list[str] = []
        if self.llm_backend == "ollama" and self.ollama_model:
            try:
                return self._extract_task_proposals_with_ollama(query, owner, retrieved), "ollama", warnings
            except Exception as exc:
                warnings.append(f"Ollama task extraction failed; used deterministic parser instead: {exc}")
        if self.llm_backend == "gemini":
            try:
                return self._extract_task_proposals_with_gemini(query, owner, retrieved), "gemini", warnings
            except Exception as exc:
                warnings.append(f"Gemini task extraction failed; used deterministic parser instead: {exc}")
        return self._extract_task_proposals(query, owner), "deterministic", warnings

    def _extract_task_proposals(self, query: str, owner: Owner) -> list[TaskProposal]:
        pet_name = self._extract_pet_name(query, owner)
        if pet_name is None:
            return []

        description = self._extract_description(query)
        due_date = self._parse_date(query).isoformat()
        due_time = self._parse_time(query).strftime("%H:%M")
        duration = self._parse_duration(query) or 30
        frequency = (
            "daily"
            if "every day" in query.lower() or "daily" in query.lower()
            else "weekly"
            if "weekly" in query.lower() or "every week" in query.lower()
            else "once"
        )
        priority = (
            "high"
            if "high priority" in query.lower() or "urgent" in query.lower()
            else "low"
            if "low priority" in query.lower()
            else "medium"
        )
        task_type = self._infer_task_type(description)

        return [
            TaskProposal(
                pet_name=pet_name,
                description=description,
                due_date=due_date,
                due_time=due_time,
                frequency=frequency,
                priority=priority,
                duration_minutes=duration,
                task_type=task_type,
            )
        ]

    def _task_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pet_name": {"type": "string"},
                            "description": {"type": "string"},
                            "due_date": {"type": "string"},
                            "due_time": {"type": "string"},
                            "frequency": {"type": "string"},
                            "priority": {"type": "string"},
                            "duration_minutes": {"type": "integer"},
                            "task_type": {"type": "string"},
                        },
                        "required": [
                            "pet_name",
                            "description",
                            "due_date",
                            "due_time",
                            "frequency",
                            "priority",
                            "duration_minutes",
                            "task_type",
                        ],
                    },
                }
            },
            "required": ["proposals"],
        }

    def _build_task_extraction_prompt(self, query: str, owner: Owner, retrieved: list[RetrievedChunk]) -> str:
        pet_names = [pet.name for pet in owner.pets]
        context_lines = [f"- {chunk.title}: {chunk.text}" for chunk in retrieved[:5]]
        today = date.today().isoformat()
        return (
            f"Today's date is {today}.\n"
            f"Allowed pet names: {', '.join(pet_names)}\n"
            "Allowed frequency values: once, daily, weekly\n"
            "Allowed priority values: low, medium, high\n"
            "Allowed task_type values: feeding, exercise, medical, grooming, appointment, general\n"
            "Use 24-hour HH:MM format and YYYY-MM-DD dates.\n"
            f"Relevant context:\n" + "\n".join(context_lines) + "\n\n"
            f"Request: {query}"
        )

    def _coerce_task_proposals(self, payload: dict[str, Any]) -> list[TaskProposal]:
        proposals: list[TaskProposal] = []
        for item in payload.get("proposals", []):
            if not isinstance(item, dict):
                continue
            try:
                proposals.append(
                    TaskProposal(
                        pet_name=str(item.get("pet_name", "")).strip(),
                        description=str(item.get("description", "")).strip(),
                        due_date=str(item.get("due_date", "")).strip(),
                        due_time=str(item.get("due_time", "")).strip(),
                        frequency=str(item.get("frequency", "once")).strip().lower(),
                        priority=str(item.get("priority", "medium")).strip().lower(),
                        duration_minutes=int(item.get("duration_minutes", 30)),
                        task_type=str(item.get("task_type", "general")).strip().lower(),
                    )
                )
            except (TypeError, ValueError):
                continue
        return proposals

    def _extract_task_proposals_with_ollama(
        self,
        query: str,
        owner: Owner,
        retrieved: list[RetrievedChunk],
    ) -> list[TaskProposal]:
        payload = self.ollama_client.chat_json(
            model=self.ollama_model,
            schema=self._task_schema(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert natural-language pet scheduling requests into valid JSON. "
                        "Use only the allowed pet names and output nothing but the schema."
                    ),
                },
                {"role": "user", "content": self._build_task_extraction_prompt(query, owner, retrieved)},
            ],
        )
        return self._coerce_task_proposals(payload)

    def _extract_task_proposals_with_gemini(
        self,
        query: str,
        owner: Owner,
        retrieved: list[RetrievedChunk],
    ) -> list[TaskProposal]:
        client = self.gemini_client or GeminiClient()
        payload = client.generate_json(
            model=self.gemini_model,
            schema=self._task_schema(),
            prompt=(
                "You convert natural-language pet scheduling requests into valid JSON. "
                "Use only the allowed pet names and output only schema-compliant JSON.\n\n"
                + self._build_task_extraction_prompt(query, owner, retrieved)
            ),
        )
        return self._coerce_task_proposals(payload)

    def _extract_pet_name(self, query: str, owner: Owner) -> Optional[str]:
        lowered = query.lower()
        for pet in owner.pets:
            if pet.name.lower() in lowered:
                return pet.name
        return owner.pets[0].name if len(owner.pets) == 1 else None

    def _extract_description(self, query: str) -> str:
        lowered = query.lower()
        patterns = [
            r"(?:schedule|add|set up|remind me to)\s+(?:a|an)?\s*(.+?)\s+for\s+[A-Za-z]+",
            r"(?:schedule|add|set up|remind me to)\s+(.+?)(?:\s+at\s+|\s+tomorrow|\s+today|\s+every\s+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return self._clean_description(match.group(1))
        if "walk" in lowered:
            return "Walk"
        if "feed" in lowered or "meal" in lowered or "breakfast" in lowered or "dinner" in lowered:
            return "Feeding"
        if "med" in lowered:
            return "Medication"
        if "groom" in lowered or "brush" in lowered:
            return "Grooming"
        return "Pet care task"

    def _clean_description(self, text: str) -> str:
        text = re.sub(r"\b(?:a|an|the)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.")
        return text.title() if text else "Pet care task"

    def _parse_duration(self, query: str) -> Optional[int]:
        match = re.search(r"(\d+)\s*(minute|minutes|min)\b", query, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _parse_time(self, query: str) -> datetime.time:
        lowered = query.lower()
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            period = match.group(3)
            if period == "pm" and hour != 12:
                hour += 12
            if period == "am" and hour == 12:
                hour = 0
            return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()

        match_24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lowered)
        if match_24:
            return datetime.strptime(f"{int(match_24.group(1)):02d}:{match_24.group(2)}", "%H:%M").time()

        if "morning" in lowered:
            return datetime.strptime("07:00", "%H:%M").time()
        if "evening" in lowered:
            return datetime.strptime("18:00", "%H:%M").time()
        if "afternoon" in lowered:
            return datetime.strptime("14:00", "%H:%M").time()
        return datetime.strptime("09:00", "%H:%M").time()

    def _parse_date(self, query: str, default_today: bool = True) -> date:
        today = date.today()
        lowered = query.lower()

        match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
        if match:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        if "tomorrow" in lowered:
            return today + timedelta(days=1)
        if "today" in lowered and default_today:
            return today

        weekdays = {name: idx for idx, name in enumerate(
            ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        )}
        for name, target in weekdays.items():
            if name in lowered:
                days_ahead = (target - today.weekday()) % 7
                return today + timedelta(days=days_ahead if days_ahead != 0 else 7)

        return today

    def _infer_task_type(self, description: str) -> str:
        lowered = description.lower()
        if any(word in lowered for word in ["feed", "meal", "breakfast", "dinner"]):
            return "feeding"
        if any(word in lowered for word in ["walk", "exercise", "play"]):
            return "exercise"
        if any(word in lowered for word in ["med", "pill", "appointment", "vet"]):
            return "medical" if "appointment" not in lowered and "vet" not in lowered else "appointment"
        if any(word in lowered for word in ["groom", "brush", "nail", "bath"]):
            return "grooming"
        return "general"

    def _compose_grounded_answer(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        del query
        if not retrieved:
            return "I could not retrieve enough context to answer that confidently."

        top_kb = [chunk for chunk in retrieved if chunk.source == "kb"][:2]
        top_schedule = [chunk for chunk in retrieved if chunk.source == "schedule"][:2]
        parts = []
        if top_schedule:
            parts.append("Schedule context:")
            for chunk in top_schedule:
                parts.append(f"- [{chunk.title}] {chunk.text}")
        if top_kb:
            parts.append("Grounded care guidance:")
            for chunk in top_kb:
                parts.append(f"- [{chunk.title}] {chunk.text}")
        if not top_kb and not top_schedule:
            parts.append("I found limited context, so keep the answer conservative and verify with your vet.")
        return "\n".join(parts)

    def _answer_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "warnings"],
        }

    def _build_answer_prompt(
        self,
        *,
        query: str,
        intent: str,
        retrieved: list[RetrievedChunk],
        draft_answer: str,
        tool_outputs: dict,
        self_critique: dict,
    ) -> str:
        context_lines = [f"- ({chunk.source}) {chunk.title}: {chunk.text}" for chunk in retrieved[:5]]
        return (
            "You are a cautious pet-care assistant. Use only the provided grounded context and tool outputs. "
            "Do not invent medical treatment, diagnoses, or schedule facts. If information is missing, say so clearly.\n\n"
            f"Intent: {intent}\n"
            f"User request: {query}\n\n"
            f"Deterministic draft answer:\n{draft_answer}\n\n"
            f"Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
            f"Tool outputs:\n{json.dumps(tool_outputs, indent=2)}\n\n"
            f"Existing critique:\n{json.dumps(self_critique, indent=2)}\n\n"
            "Return a concise grounded answer and any brief caveats as JSON."
        )

    def _synthesize_answer_with_model(
        self,
        *,
        query: str,
        intent: str,
        retrieved: list[RetrievedChunk],
        draft_answer: str,
        tool_outputs: dict,
        self_critique: dict,
    ) -> tuple[str, list[str], str]:
        schema = self._answer_schema()
        prompt = self._build_answer_prompt(
            query=query,
            intent=intent,
            retrieved=retrieved,
            draft_answer=draft_answer,
            tool_outputs=tool_outputs,
            self_critique=self_critique,
        )
        if self.llm_backend == "ollama" and self.ollama_model:
            payload = self.ollama_client.chat_json(
                model=self.ollama_model,
                schema=schema,
                messages=[{"role": "user", "content": prompt}],
            )
            model_used = self.ollama_model
        elif self.llm_backend == "gemini":
            client = self.gemini_client or GeminiClient()
            payload = client.generate_json(model=self.gemini_model, prompt=prompt, schema=schema)
            model_used = self.gemini_model
        else:
            return draft_answer, [], "deterministic"
        answer = str(payload.get("answer", "")).strip() or draft_answer
        warnings = [str(item) for item in payload.get("warnings", []) if str(item).strip()]
        return answer, warnings, model_used

    def _summarize_task_proposals(self, proposals: list[TaskProposal]) -> str:
        lines = ["I parsed the following task proposal(s):"]
        for proposal in proposals:
            lines.append(
                f"- {proposal.pet_name}: {proposal.description} on {proposal.due_date} at "
                f"{proposal.due_time}, {proposal.duration_minutes} minutes, {proposal.frequency}, "
                f"{proposal.priority} priority"
            )
        lines.append("Approve them in the UI or CLI to add them to the schedule.")
        return "\n".join(lines)

    def _format_schedule_review(self, summary: Any) -> str:
        lines = [
            f"Pending tasks: {summary.pending_count}",
            f"Completed tasks: {summary.completed_count}",
            f"High-priority pending tasks: {summary.high_priority_count}",
        ]
        if summary.next_tasks:
            lines.append("Next tasks:")
            for item in summary.next_tasks:
                lines.append(
                    f"- {item['pet']}: {item['task']} on {item['date']} at {item['time']} ({item['priority']})"
                )
        if summary.conflicts:
            lines.append("Conflicts:")
            for warning in summary.conflicts:
                lines.append(f"- {warning}")
        else:
            lines.append("No conflicts detected.")
        return "\n".join(lines)

    def _score_confidence(
        self,
        intent: str,
        query: str,
        retrieved: list[RetrievedChunk],
        proposals: list[TaskProposal],
        critique: dict,
    ) -> dict:
        score = 0.45
        basis = []
        if retrieved:
            score += 0.20
            basis.append("retrieved_context")
        if any(chunk.source == "schedule" for chunk in retrieved):
            score += 0.10
            basis.append("schedule_grounding")
        if intent == "add_task" and proposals:
            score += 0.15
            basis.append("structured_task_extraction")
        if critique.get("warnings"):
            score -= 0.10
            basis.append("warnings_present")
        if self._should_use_model():
            score += 0.05
            basis.append("model_synthesis")
        if len(query.split()) >= 5:
            score += 0.05
            basis.append("sufficient_query_detail")
        return {"score": max(0.05, min(0.99, round(score, 2))), "basis": basis}

    def _step(self, step: int, thought: str, action: str, observation: str) -> AgentTraceStep:
        return AgentTraceStep(step=step, thought=thought, action=action, observation=observation)

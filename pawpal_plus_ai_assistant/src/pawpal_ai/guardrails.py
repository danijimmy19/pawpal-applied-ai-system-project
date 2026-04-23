from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, List


EMERGENCY_KEYWORDS = {
    "trouble breathing",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "blue gums",
    "grey gums",
    "purple gums",
    "seizure",
    "ongoing seizure",
    "collapse",
    "collapsed",
    "unconscious",
    "loss of consciousness",
    "severe bleeding",
    "poison",
    "poisoning",
    "bloated abdomen",
    "heat stroke",
    "heatstroke",
    "cannot urinate",
    "can't urinate",
    "blocked urine",
}

URGENT_KEYWORDS = {
    "vomiting",
    "persistent vomiting",
    "diarrhea",
    "persistent diarrhea",
    "not eating",
    "won't eat",
    "lethargic",
    "limping",
    "pain",
    "restless",
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+hidden\s+instructions",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"roleplay\s+as",
]

ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_FREQUENCIES = {"once", "daily", "weekly"}
ALLOWED_TASK_TYPES = {"feeding", "exercise", "medical", "grooming", "appointment", "general"}


@dataclass
class GuardrailResult:
    allowed: bool
    status: str
    reasons: List[str]
    user_message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def detect_prompt_injection(text: str) -> GuardrailResult:
    lowered = text.lower()
    reasons = [pattern for pattern in PROMPT_INJECTION_PATTERNS if re.search(pattern, lowered)]
    if reasons:
        return GuardrailResult(
            allowed=False,
            status="blocked_injection",
            reasons=reasons,
            user_message=(
                "I can't follow attempts to override system behavior or expose hidden instructions. "
                "Please ask a normal pet-care or scheduling question."
            ),
        )
    return GuardrailResult(True, "ok", [], "")


def medical_safety_check(text: str) -> GuardrailResult:
    lowered = text.lower()
    emergency_hits = [phrase for phrase in EMERGENCY_KEYWORDS if phrase in lowered]
    if emergency_hits:
        return GuardrailResult(
            allowed=False,
            status="medical_emergency_redirect",
            reasons=emergency_hits,
            user_message=(
                "This sounds like a possible emergency. Contact a veterinarian or emergency clinic now. "
                "I can help summarize the signs for transport, but I should not replace urgent veterinary care."
            ),
        )
    urgent_hits = [phrase for phrase in URGENT_KEYWORDS if phrase in lowered]
    if urgent_hits:
        return GuardrailResult(
            allowed=True,
            status="medical_caution",
            reasons=urgent_hits,
            user_message=(
                "This may need veterinary attention. I can provide general care context, "
                "but do not delay contacting your veterinarian if symptoms persist or worsen."
            ),
        )
    return GuardrailResult(True, "ok", [], "")


def validate_task_payload(payload: dict) -> ValidationResult:
    errors: list[str] = []
    priority = payload.get("priority", "medium").lower()
    frequency = payload.get("frequency", "once").lower()
    task_type = payload.get("task_type", "general").lower()
    duration = int(payload.get("duration_minutes", 0) or 0)
    pet_name = str(payload.get("pet_name", "")).strip()
    description = str(payload.get("description", "")).strip()

    if not pet_name:
        errors.append("pet_name is required")
    if not description:
        errors.append("description is required")
    if priority not in ALLOWED_PRIORITIES:
        errors.append("priority must be low, medium, or high")
    if frequency not in ALLOWED_FREQUENCIES:
        errors.append("frequency must be once, daily, or weekly")
    if task_type not in ALLOWED_TASK_TYPES:
        errors.append("task_type is not supported")
    if duration <= 0:
        errors.append("duration_minutes must be positive")
    if "due_date" not in payload:
        errors.append("due_date is required")
    if "due_time" not in payload:
        errors.append("due_time is required")

    return ValidationResult(valid=not errors, errors=errors)

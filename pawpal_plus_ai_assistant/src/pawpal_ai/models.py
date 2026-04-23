from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional
import json


_PRIORITY_WEIGHTS = {"high": 0, "medium": 1, "low": 2}
_PRIORITY_EMOJIS = {"high": "🔴", "medium": "🟡", "low": "🟢"}


@dataclass
class Task:
    """A single pet-care task."""

    description: str
    due_date: date
    due_time: time
    frequency: str = "once"
    priority: str = "medium"
    completed: bool = False
    duration_minutes: int = 30
    task_type: str = "general"

    def __post_init__(self) -> None:
        self.priority = self.priority.lower()
        self.frequency = self.frequency.lower()
        self.task_type = self.task_type.lower()
        if self.priority not in _PRIORITY_WEIGHTS:
            raise ValueError("priority must be low, medium, or high")
        if self.frequency not in {"once", "daily", "weekly"}:
            raise ValueError("frequency must be once, daily, or weekly")
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")

    @property
    def due_datetime(self) -> datetime:
        return datetime.combine(self.due_date, self.due_time)

    def mark_complete(self) -> None:
        self.completed = True

    def next_occurrence(self) -> Optional["Task"]:
        if self.frequency == "once":
            return None
        delta = timedelta(days=1 if self.frequency == "daily" else 7)
        return Task(
            description=self.description,
            due_date=self.due_date + delta,
            due_time=self.due_time,
            frequency=self.frequency,
            priority=self.priority,
            completed=False,
            duration_minutes=self.duration_minutes,
            task_type=self.task_type,
        )

    def priority_weight(self) -> int:
        return _PRIORITY_WEIGHTS[self.priority]

    def formatted_status(self) -> str:
        return "✅ Complete" if self.completed else "⏳ Pending"

    def formatted_priority(self) -> str:
        return f"{_PRIORITY_EMOJIS[self.priority]} {self.priority.title()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "due_date": self.due_date.isoformat(),
            "due_time": self.due_time.strftime("%H:%M"),
            "frequency": self.frequency,
            "priority": self.priority,
            "completed": self.completed,
            "duration_minutes": self.duration_minutes,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            description=data["description"],
            due_date=date.fromisoformat(data["due_date"]),
            due_time=datetime.strptime(data["due_time"], "%H:%M").time(),
            frequency=data.get("frequency", "once"),
            priority=data.get("priority", "medium"),
            completed=data.get("completed", False),
            duration_minutes=int(data.get("duration_minutes", 30)),
            task_type=data.get("task_type", "general"),
        )


@dataclass
class Pet:
    name: str
    species: str
    age: int
    notes: str = ""
    tasks: List[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def list_tasks(self, include_completed: bool = True) -> List[Task]:
        if include_completed:
            return list(self.tasks)
        return [task for task in self.tasks if not task.completed]

    def task_count(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "species": self.species,
            "age": self.age,
            "notes": self.notes,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pet":
        pet = cls(
            name=data["name"],
            species=data["species"],
            age=int(data.get("age", 0)),
            notes=data.get("notes", ""),
        )
        pet.tasks = [Task.from_dict(task_data) for task_data in data.get("tasks", [])]
        return pet


@dataclass
class Owner:
    name: str
    email: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)
    pets: List[Pet] = field(default_factory=list)

    def add_pet(self, pet: Pet) -> None:
        self.pets.append(pet)

    def get_pet(self, pet_name: str) -> Optional[Pet]:
        normalized = pet_name.strip().lower()
        for pet in self.pets:
            if pet.name.lower() == normalized:
                return pet
        return None

    def all_tasks(self) -> List[tuple[Pet, Task]]:
        return [(pet, task) for pet in self.pets for task in pet.tasks]

    def save_to_json(self, filepath: str) -> None:
        payload = {
            "name": self.name,
            "email": self.email,
            "preferences": self.preferences,
            "pets": [pet.to_dict() for pet in self.pets],
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Owner":
        with open(filepath, "r", encoding="utf-8") as file:
            payload = json.load(file)
        owner = cls(
            name=payload["name"],
            email=payload.get("email", ""),
            preferences=payload.get("preferences", {}),
        )
        owner.pets = [Pet.from_dict(pet_data) for pet_data in payload.get("pets", [])]
        return owner

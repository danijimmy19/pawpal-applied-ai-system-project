from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

from .models import Owner, Pet, Task


@dataclass
class ScheduleSummary:
    pending_count: int
    completed_count: int
    high_priority_count: int
    conflicts: List[str]
    next_tasks: List[Dict[str, str]]


class Scheduler:
    """Coordinates task retrieval, sorting, filtering, and scheduling logic."""

    def __init__(self, owner: Owner) -> None:
        self.owner = owner

    def get_all_tasks(self, include_completed: bool = True) -> List[tuple[Pet, Task]]:
        tasks = self.owner.all_tasks()
        if include_completed:
            return tasks
        return [(pet, task) for pet, task in tasks if not task.completed]

    def sort_tasks_by_time(self, include_completed: bool = True) -> List[tuple[Pet, Task]]:
        return sorted(self.get_all_tasks(include_completed), key=lambda item: item[1].due_datetime)

    def sort_by_priority_then_time(
        self, include_completed: bool = True
    ) -> List[tuple[Pet, Task]]:
        return sorted(
            self.get_all_tasks(include_completed),
            key=lambda item: (item[1].priority_weight(), item[1].due_datetime),
        )

    def filter_tasks(
        self,
        *,
        pet_name: Optional[str] = None,
        completed: Optional[bool] = None,
        priority: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[tuple[Pet, Task]]:
        filtered = self.get_all_tasks(include_completed=True)
        if pet_name is not None:
            normalized = pet_name.strip().lower()
            filtered = [(pet, task) for pet, task in filtered if pet.name.lower() == normalized]
        if completed is not None:
            filtered = [(pet, task) for pet, task in filtered if task.completed == completed]
        if priority is not None:
            normalized = priority.strip().lower()
            filtered = [(pet, task) for pet, task in filtered if task.priority == normalized]
        if task_type is not None:
            normalized = task_type.strip().lower()
            filtered = [(pet, task) for pet, task in filtered if task.task_type == normalized]
        return filtered

    def detect_conflicts(self) -> List[str]:
        warnings: List[str] = []
        ordered = self.sort_tasks_by_time(include_completed=False)
        for index, (pet_a, task_a) in enumerate(ordered):
            start_a = task_a.due_datetime
            end_a = start_a + timedelta(minutes=task_a.duration_minutes)
            for pet_b, task_b in ordered[index + 1 :]:
                start_b = task_b.due_datetime
                if start_b > end_a:
                    break
                end_b = start_b + timedelta(minutes=task_b.duration_minutes)
                overlaps = max(start_a, start_b) < min(end_a, end_b)
                if overlaps:
                    warnings.append(
                        f"Conflict: {pet_a.name} '{task_a.description}' overlaps with "
                        f"{pet_b.name} '{task_b.description}' at {start_a.strftime('%Y-%m-%d %H:%M')}"
                    )
        return warnings

    def mark_task_complete(self, pet_name: str, description: str) -> Optional[Task]:
        pet = self.owner.get_pet(pet_name)
        if pet is None:
            return None
        for task in pet.tasks:
            if task.description.lower() == description.lower() and not task.completed:
                task.mark_complete()
                next_task = task.next_occurrence()
                if next_task is not None:
                    pet.add_task(next_task)
                return task
        return None

    def next_available_slot(
        self,
        on_date: date,
        duration_minutes: int,
        start_hour: int = 8,
        end_hour: int = 20,
    ) -> Optional[time]:
        if duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")

        existing = []
        for _, task in self.get_all_tasks(include_completed=False):
            if task.due_date == on_date:
                start_dt = task.due_datetime
                end_dt = start_dt + timedelta(minutes=task.duration_minutes)
                existing.append((start_dt, end_dt))
        existing.sort(key=lambda item: item[0])

        cursor = datetime.combine(on_date, time(start_hour, 0))
        day_end = datetime.combine(on_date, time(end_hour, 0))
        if not existing and cursor + timedelta(minutes=duration_minutes) <= day_end:
            return cursor.time()

        for start_dt, end_dt in existing:
            if cursor + timedelta(minutes=duration_minutes) <= start_dt:
                return cursor.time()
            if end_dt > cursor:
                cursor = end_dt

        if cursor + timedelta(minutes=duration_minutes) <= day_end:
            return cursor.time()
        return None

    def add_task_to_pet(self, pet_name: str, task: Task) -> bool:
        pet = self.owner.get_pet(pet_name)
        if pet is None:
            return False
        pet.add_task(task)
        return True

    def agenda_table(self, sort_mode: str = "time", include_completed: bool = True) -> List[Dict[str, str]]:
        if sort_mode == "priority":
            ordered = self.sort_by_priority_then_time(include_completed)
        else:
            ordered = self.sort_tasks_by_time(include_completed)

        rows = []
        for pet, task in ordered:
            rows.append(
                {
                    "Pet": pet.name,
                    "Species": pet.species,
                    "Task": task.description,
                    "Type": task.task_type.title(),
                    "Date": task.due_date.isoformat(),
                    "Time": task.due_time.strftime("%H:%M"),
                    "Duration": f"{task.duration_minutes} min",
                    "Priority": task.formatted_priority(),
                    "Status": task.formatted_status(),
                    "Frequency": task.frequency.title(),
                }
            )
        return rows

    def summarize_schedule(self, on_date: Optional[date] = None) -> ScheduleSummary:
        rows = self.sort_by_priority_then_time(include_completed=True)
        if on_date is not None:
            rows = [(pet, task) for pet, task in rows if task.due_date == on_date]
        pending = [(pet, task) for pet, task in rows if not task.completed]
        completed = [(pet, task) for pet, task in rows if task.completed]
        next_tasks = []
        for pet, task in pending[:5]:
            next_tasks.append(
                {
                    "pet": pet.name,
                    "task": task.description,
                    "date": task.due_date.isoformat(),
                    "time": task.due_time.strftime("%H:%M"),
                    "priority": task.priority,
                }
            )
        return ScheduleSummary(
            pending_count=len(pending),
            completed_count=len(completed),
            high_priority_count=sum(1 for _, task in pending if task.priority == "high"),
            conflicts=self.detect_conflicts(),
            next_tasks=next_tasks,
        )

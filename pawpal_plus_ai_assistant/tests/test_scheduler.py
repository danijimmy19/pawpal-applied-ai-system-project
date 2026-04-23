from datetime import date, time, timedelta

from pawpal_ai.models import Owner, Pet, Task
from pawpal_ai.scheduler import Scheduler


def build_scheduler() -> Scheduler:
    owner = Owner(name="Taylor")
    dog = Pet(name="Milo", species="Dog", age=5)
    cat = Pet(name="Nori", species="Cat", age=3)
    owner.add_pet(dog)
    owner.add_pet(cat)

    today = date.today()
    dog.add_task(Task("Walk", today, time(9, 0), frequency="daily", duration_minutes=30))
    dog.add_task(Task("Breakfast", today, time(8, 0), priority="high", duration_minutes=10, task_type="feeding"))
    cat.add_task(Task("Medication", today, time(8, 20), priority="high", duration_minutes=10, task_type="medical"))
    return Scheduler(owner)


def test_sorting_returns_chronological_order() -> None:
    scheduler = build_scheduler()
    ordered = scheduler.sort_tasks_by_time()
    assert [task.description for _, task in ordered] == ["Breakfast", "Medication", "Walk"]


def test_conflict_detection_handles_overlap() -> None:
    owner = Owner(name="Jamie")
    dog = Pet(name="Pepper", species="Dog", age=6)
    cat = Pet(name="Olive", species="Cat", age=4)
    owner.add_pet(dog)
    owner.add_pet(cat)

    same_day = date.today()
    dog.add_task(Task("Walk", same_day, time(10, 0), duration_minutes=60))
    cat.add_task(Task("Medication", same_day, time(10, 30), duration_minutes=15))

    scheduler = Scheduler(owner)
    conflicts = scheduler.detect_conflicts()
    assert len(conflicts) == 1
    assert "overlaps" in conflicts[0]


def test_daily_completion_creates_next_occurrence() -> None:
    scheduler = build_scheduler()
    updated = scheduler.mark_task_complete("Milo", "Walk")
    assert updated is not None
    milo_tasks = scheduler.filter_tasks(pet_name="Milo")
    future = [task for _, task in milo_tasks if not task.completed and task.description == "Walk"]
    assert future
    assert future[0].due_date == date.today() + timedelta(days=1)


def test_next_available_slot_skips_booked_windows() -> None:
    scheduler = build_scheduler()
    slot = scheduler.next_available_slot(date.today(), duration_minutes=20, start_hour=8, end_hour=12)
    assert slot.strftime("%H:%M") == "08:30"

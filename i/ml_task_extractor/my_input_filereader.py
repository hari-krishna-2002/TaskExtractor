# my_input_filereader.py

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from calendar_helper import create_event
from main import TaskExtractionCLI
from pathlib import Path

def display_tasks(tasks):
    """Print extracted tasks in formatted style."""
    for i, task in enumerate(tasks, 1):
        print(f"Task {i}:")
        print(f"  Task     : {task.cleaned_text}")
        print(f"  Due Date : {task.due_date}")
        print(f"  Priority : {task.priority.value}")
        print(f"  Category : {task.category.value}")
        print(f"  Original : {task.original_text}")
        print("-" * 50)

if __name__ == "__main__":
    file_path = 'textfile.txt'

    if not Path(file_path).exists():
        print(f"❌ Error: File '{file_path}' not found.")
        exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    extractor = TaskExtractionCLI(config_path=str(config_path))

    result = extractor.extract_from_text(text)

    display_tasks(result.tasks)

    for task in result.tasks:
        if task.due_date:
            print(f"Creating event: {task.cleaned_text} on {task.due_date}")
            create_event(
                task_text=task.cleaned_text,
                due_date=task.due_date.strftime('%Y-%m-%d'),
                description=f"Original: {task.original_text} | Priority: {task.priority.value} | Category: {task.category.value}"
            )
        else:
            print(f"⚠️ Skipping task '{task.cleaned_text}' (no due date found)")

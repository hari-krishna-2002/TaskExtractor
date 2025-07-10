from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime
import yaml
from pathlib import Path

from ..models.task_model import Task, MeetingTranscript, TranscriptSegment, ExtractionResult, Speaker
from ..utils.task_extractor import MLTaskExtractor
from loguru import logger

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="ML Task Extractor API",
    description="Real-time task extraction from meeting transcripts using advanced NLP",
    version="1.0.0"
)

# Add CORS middleware
if config.get('api', {}).get('enable_cors', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize task extractor
task_extractor = MLTaskExtractor(config)

# In-memory storage (in production, use a proper database)
meetings_db: Dict[str, MeetingTranscript] = {}
tasks_db: Dict[str, Task] = {}


# Request/Response models
class TranscriptProcessRequest(BaseModel):
    text: str
    speaker_name: Optional[str] = None
    speaker_role: Optional[str] = None
    meeting_context: Optional[str] = None
    meeting_id: Optional[str] = None


class RealTimeTranscriptRequest(BaseModel):
    meeting_id: str
    segment: TranscriptSegment


class TaskUpdateRequest(BaseModel):
    task_id: str
    updates: Dict[str, Any]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "ML Task Extractor API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.post("/extract-tasks", response_model=ExtractionResult)
async def extract_tasks(request: TranscriptProcessRequest):
    """Extract tasks from a transcript text."""
    try:
        start_time = datetime.utcnow()

        # Create speaker info if provided
        speaker = None
        if request.speaker_name:
            speaker = Speaker(
                name=request.speaker_name,
                role=request.speaker_role,
                authority_level=1.5 if request.speaker_role in ['manager', 'lead', 'director'] else 1.0
            )

        # Extract tasks
        tasks = task_extractor.extract_tasks_from_transcript(
            request.text,
            speaker_info=speaker,
            meeting_context=request.meeting_context
        )

        # Store tasks in database
        for task in tasks:
            tasks_db[task.id] = task

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create result
        result = ExtractionResult(
            tasks=tasks,
            processing_time=processing_time,
            total_sentences=len(request.text.split('.')),
            successful_extractions=len(tasks),
            confidence_scores=[task.confidence for task in tasks]
        )

        logger.info(f"Extracted {len(tasks)} tasks in {processing_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error extracting tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/meetings", response_model=MeetingTranscript)
async def create_meeting(title: Optional[str] = None, participants: List[Speaker] = []):
    """Create a new meeting session."""
    meeting_id = str(uuid.uuid4())
    meeting = MeetingTranscript(
        id=meeting_id,
        title=title or f"Meeting {meeting_id[:8]}",
        participants=participants,
        start_time=datetime.utcnow()
    )

    meetings_db[meeting_id] = meeting
    logger.info(f"Created meeting: {meeting_id}")
    return meeting


@app.post("/meetings/{meeting_id}/transcript")
async def add_transcript_segment(meeting_id: str, segment: TranscriptSegment):
    """Add a real-time transcript segment to a meeting."""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting = meetings_db[meeting_id]
    meeting.transcript_segments.append(segment.dict())

    # Process segment for tasks if it's substantial enough
    if len(segment.text.split()) > 5:  # Only process segments with more than 5 words
        try:
            tasks = task_extractor.extract_tasks_from_transcript(
                segment.text,
                speaker_info=segment.speaker,
                meeting_context=meeting.title
            )

            # Add tasks to meeting and global storage
            for task in tasks:
                task.meeting_context = meeting_id
                meeting.extracted_tasks.append(task)
                tasks_db[task.id] = task

            logger.info(f"Added segment to meeting {meeting_id}, extracted {len(tasks)} tasks")

        except Exception as e:
            logger.error(f"Error processing transcript segment: {e}")

    return {"status": "success", "tasks_extracted": len(meeting.extracted_tasks)}


@app.get("/meetings/{meeting_id}", response_model=MeetingTranscript)
async def get_meeting(meeting_id: str):
    """Get meeting details and extracted tasks."""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return meetings_db[meeting_id]


@app.get("/meetings/{meeting_id}/tasks", response_model=List[Task])
async def get_meeting_tasks(meeting_id: str):
    """Get all tasks extracted from a specific meeting."""
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting_tasks = [task for task in tasks_db.values() if task.meeting_context == meeting_id]
    return meeting_tasks


@app.get("/tasks", response_model=List[Task])
async def get_all_tasks(
        category: Optional[str] = None,
        priority: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
):
    """Get all tasks with optional filtering."""
    tasks = list(tasks_db.values())

    # Apply filters
    if category:
        tasks = [task for task in tasks if task.category.value == category]
    if priority:
        tasks = [task for task in tasks if task.priority.value == priority]
    if status:
        tasks = [task for task in tasks if task.status.value == status]

    # Sort by due date and confidence
    tasks.sort(key=lambda t: (t.due_date or datetime.max.date(), -t.confidence))

    return tasks[:limit]


@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]


@app.put("/tasks/{task_id}")
async def update_task(task_id: str, updates: TaskUpdateRequest):
    """Update a task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_db[task_id]

    # Update allowed fields
    for field, value in updates.updates.items():
        if hasattr(task, field):
            setattr(task, field, value)

    logger.info(f"Updated task {task_id}")
    return {"status": "success"}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    del tasks_db[task_id]
    logger.info(f"Deleted task {task_id}")
    return {"status": "success"}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    total_tasks = len(tasks_db)
    total_meetings = len(meetings_db)

    # Task statistics
    task_by_priority = {}
    task_by_category = {}
    task_by_status = {}

    for task in tasks_db.values():
        # Priority stats
        priority = task.priority.value
        task_by_priority[priority] = task_by_priority.get(priority, 0) + 1

        # Category stats
        category = task.category.value
        task_by_category[category] = task_by_category.get(category, 0) + 1

        # Status stats
        status = task.status.value
        task_by_status[status] = task_by_status.get(status, 0) + 1

    return {
        "total_tasks": total_tasks,
        "total_meetings": total_meetings,
        "task_by_priority": task_by_priority,
        "task_by_category": task_by_category,
        "task_by_status": task_by_status,
        "average_confidence": sum(task.confidence for task in tasks_db.values()) / total_tasks if total_tasks > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn

    host = config.get('api', {}).get('host', '0.0.0.0')
    port = config.get('api', {}).get('port', 8000)

    uvicorn.run(app, host=host, port=port)
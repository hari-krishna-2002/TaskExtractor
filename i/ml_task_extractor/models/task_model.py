from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, time
from enum import Enum
from uuid import UUID


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    WORK = "work"
    PERSONAL = "personal"
    ADMINISTRATIVE = "administrative"
    TECHNICAL = "technical"
    MEETING = "meeting"
    DEADLINE = "deadline"
    UNCATEGORIZED = "uncategorized"


class ExtractionMethod(str, Enum):
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    INFERRED = "inferred"
    ML_PREDICTED = "ml_predicted"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Speaker(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    authority_level: float = Field(default=1.0, ge=0.0, le=2.0)


class Task(BaseModel):
    id: str = Field(..., description="Unique task identifier")
    original_text: str = Field(..., description="Original sentence from transcript")
    cleaned_text: str = Field(..., description="Cleaned task description")
    contextual_text: Optional[str] = Field(None, description="Context-resolved task text")

    # Core task properties
    due_date: Optional[date] = None
    due_time: Optional[str] = Field(None, pattern=r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$")
    priority: Priority = Priority.MEDIUM
    category: Category = Category.UNCATEGORIZED
    status: TaskStatus = TaskStatus.PENDING

    # Extraction metadata
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_method: ExtractionMethod = ExtractionMethod.DIRECT
    requires_review: bool = False

    # Context information
    speaker: Optional[Speaker] = None
    meeting_context: Optional[str] = None
    related_tasks: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    # Timestamps
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    mentioned_at: Optional[datetime] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeetingTranscript(BaseModel):
    id: str
    title: Optional[str] = None
    participants: List[Speaker] = Field(default_factory=list)
    start_time: datetime
    end_time: Optional[datetime] = None
    transcript_segments: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_tasks: List[Task] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    timestamp: datetime
    speaker: Optional[Speaker] = None
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_processed: bool = False


class ExtractionResult(BaseModel):
    tasks: List[Task]
    processing_time: float
    total_sentences: int
    successful_extractions: int
    confidence_scores: List[float]
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
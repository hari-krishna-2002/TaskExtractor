"""
ML Task Extractor - Advanced task extraction from meeting transcripts

A comprehensive machine learning system for extracting actionable tasks
from real-time meeting transcripts using state-of-the-art NLP techniques.
"""

__version__ = "1.0.0"
__author__ = "ML Task Extractor Team"
__email__ = "contact@mltaskextractor.com"

from .models.task_model import Task, Priority, Category, ExtractionMethod, TaskStatus, Speaker
from .utils.task_extractor import MLTaskExtractor
from .utils.nlp_processor import NLPProcessor
from .utils.date_processor import DateProcessor

__all__ = [
    "Task",
    "Priority",
    "Category",
    "ExtractionMethod",
    "TaskStatus",
    "Speaker",
    "MLTaskExtractor",
    "NLPProcessor",
    "DateProcessor"
]
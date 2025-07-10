#!/usr/bin/env python3
"""
ML Task Extractor - Main CLI Interface
Real-time task extraction from meeting transcripts using advanced NLP
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml
from loguru import logger

from models.task_model import Task as PydanticTask

from models.task_model import Task, Speaker, ExtractionResult
from ml_task_extractor.utils.task_extractor import MLTaskExtractor
from ml_task_extractor.utils.nlp_processor import NLPProcessor
from ml_task_extractor.utils.date_processor import DateProcessor


class TaskExtractionCLI:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the CLI with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.task_extractor = MLTaskExtractor(self.config)

        # Setup logging
        log_config = self.config.get('logging', {})
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '{time} | {level} | {message}')
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def extract_from_text(self, text: str, speaker_name: str = None,
                          speaker_role: str = None, meeting_context: str = None) -> ExtractionResult:
        """Extract tasks from text input."""
        logger.info("Starting task extraction from text input")

        # Create speaker info if provided
        speaker = None
        if speaker_name:
            speaker = Speaker(
                name=speaker_name,
                role=speaker_role,
                authority_level=1.5 if speaker_role in ['manager', 'lead', 'director'] else 1.0
            )

        # Extract tasks
        start_time = asyncio.get_event_loop().time()
        tasks = self.task_extractor.extract_tasks_from_transcript(
            text, speaker_info=speaker, meeting_context=meeting_context
        )
        processing_time = asyncio.get_event_loop().time() - start_time

        # Create result
        result = ExtractionResult(
            tasks=[PydanticTask.model_validate(task.__dict__) for task in tasks],  # ✅ Safely convert
            processing_time=processing_time,
            total_sentences=len(text.split('.')),
            successful_extractions=len(tasks),
            confidence_scores=[task.confidence for task in tasks]
        )

        logger.info(f"Extracted {len(tasks)} tasks in {processing_time:.2f}s")
        return result

    def extract_from_file(self, file_path: str, **kwargs) -> ExtractionResult:
        """Extract tasks from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            logger.info(f"Processing file: {file_path}")
            return self.extract_from_text(text, **kwargs)

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            sys.exit(1)

    def print_results(self, result: ExtractionResult, output_format: str = "text"):
        """Print extraction results in specified format."""
        if output_format == "json":
            # Convert to JSON-serializable format
            tasks_dict = []
            for task in result.tasks:
                task_dict = task.dict()
                # Convert date objects to strings
                if task_dict['due_date']:
                    task_dict['due_date'] = task_dict['due_date'].isoformat()
                if task_dict['extracted_at']:
                    task_dict['extracted_at'] = task_dict['extracted_at'].isoformat()
                if task_dict['mentioned_at']:
                    task_dict['mentioned_at'] = task_dict['mentioned_at'].isoformat()
                tasks_dict.append(task_dict)

            output = {
                "tasks": tasks_dict,
                "summary": {
                    "total_tasks": len(result.tasks),
                    "processing_time": result.processing_time,
                    "average_confidence": sum(result.confidence_scores) / len(
                        result.confidence_scores) if result.confidence_scores else 0
                }
            }
            print(json.dumps(output, indent=2))

        else:  # text format
            print(f"\n{'=' * 60}")
            print(f"TASK EXTRACTION RESULTS")
            print(f"{'=' * 60}")
            print(f"Total tasks extracted: {len(result.tasks)}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(
                f"Average confidence: {sum(result.confidence_scores) / len(result.confidence_scores) if result.confidence_scores else 0:.2f}")
            print(f"{'=' * 60}\n")

            for i, task in enumerate(result.tasks, 1):
                print(f"TASK #{i}")
                print(f"{'─' * 40}")
                print(f"Text: {task.cleaned_text}")
                print(f"Original: {task.original_text}")
                if task.contextual_text:
                    print(f"Contextual: {task.contextual_text}")
                print(f"Due Date: {task.due_date or 'Not specified'}")
                if task.due_time:
                    print(f"Due Time: {task.due_time}")
                print(f"Priority: {task.priority.value.upper()}")
                print(f"Category: {task.category.value.title()}")
                print(f"Status: {task.status.value.title()}")
                print(f"Confidence: {task.confidence:.2f}")
                print(f"Method: {task.extraction_method.value.title()}")
                if task.requires_review:
                    print(f"⚠️  Requires Review")
                if task.speaker:
                    print(f"Speaker: {task.speaker.name} ({task.speaker.role or 'Unknown role'})")
                if task.keywords:
                    print(f"Keywords: {', '.join(task.keywords)}")
                print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Task Extractor - Extract tasks from meeting transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from text input
  python main.py --text "We need to submit the report by tomorrow"

  # Extract from file
  python main.py --file meeting_transcript.txt

  # With speaker information
  python main.py --file transcript.txt --speaker-name "John Doe" --speaker-role "manager"

  # JSON output
  python main.py --text "Complete the project by Friday" --format json

  # Start API server
  python main.py --api
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', '-t', help='Text to extract tasks from')
    input_group.add_argument('--file', '-f', help='File containing text to process')
    input_group.add_argument('--api', action='store_true', help='Start API server')

    # Speaker information
    parser.add_argument('--speaker-name', help='Name of the speaker')
    parser.add_argument('--speaker-role', help='Role of the speaker')
    parser.add_argument('--meeting-context', help='Meeting context/title')

    # Output options
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Configuration file path')

    args = parser.parse_args()

    # Initialize CLI
    cli = TaskExtractionCLI(args.config)

    if args.api:
        # Start API server
        logger.info("Starting API server...")
        from api.main import app
        import uvicorn

        host = cli.config.get('api', {}).get('host', '0.0.0.0')
        port = cli.config.get('api', {}).get('port', 8000)

        uvicorn.run(app, host=host, port=port)

    else:
        # Process text or file
        if args.text:
            result = cli.extract_from_text(
                args.text,
                speaker_name=args.speaker_name,
                speaker_role=args.speaker_role,
                meeting_context=args.meeting_context
            )
        else:
            result = cli.extract_from_file(
                args.file,
                speaker_name=args.speaker_name,
                speaker_role=args.speaker_role,
                meeting_context=args.meeting_context
            )

        # Print results
        cli.print_results(result, args.format)


if __name__ == "__main__":
    main()
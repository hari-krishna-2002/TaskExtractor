import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import re
from loguru import logger

from ..models.task_model import Task, Priority, Category, ExtractionMethod, TaskStatus, Speaker
from .nlp_processor import NLPProcessor
from .date_processor import DateProcessor


class MLTaskExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.nlp_processor = NLPProcessor(config.get("task_extraction", {}))

        self.date_processor = DateProcessor(config)

        # Enhanced keyword sets for meeting contexts
        self.action_verbs = {
            "submit", "complete", "finish", "send", "deliver", "review", "prepare",
            "create", "update", "call", "email", "schedule", "book", "reserve",
            "order", "buy", "purchase", "pay", "attend", "meet", "discuss", "plan",
            "organize", "arrange", "confirm", "follow up", "check", "verify", "test",
            "implement", "develop", "design", "analyze", "research", "coordinate",
            "present", "demonstrate", "train", "document", "approve", "sign"
        }

        self.task_indicators = {
            "task", "assignment", "project", "deadline", "due", "reminder",
            "appointment", "meeting", "report", "presentation", "document",
            "proposal", "invoice", "quote", "contract", "action item",
            "deliverable", "milestone", "objective", "goal", "requirement"
        }

        self.meeting_specific_patterns = [
            r'\b(action\s+item|ai)\b',
            r'\b(follow\s+up|followup)\b',
            r'\b(next\s+steps?)\b',
            r'\b(to\s+do|todo)\b',
            r'\b(assigned?\s+to)\b',
            r'\b(responsible\s+for)\b',
            r'\b(will\s+handle|will\s+take\s+care\s+of)\b',
            r'\b(needs?\s+to\s+be\s+done)\b',
            r'\b(should\s+be\s+completed?)\b',
        ]

        self.priority_indicators = {
            Priority.URGENT: ["urgent", "asap", "immediately", "critical", "emergency"],
            Priority.HIGH: ["important", "priority", "soon", "quickly", "deadline"],
            Priority.MEDIUM: ["should", "need to", "required", "necessary"],
            Priority.LOW: ["maybe", "eventually", "when possible", "consider", "optional"]
        }

        self.category_keywords = {
            Category.WORK: ["project", "report", "presentation", "client", "business", "meeting"],
            Category.TECHNICAL: ["develop", "code", "test", "deploy", "debug", "implement", "system"],
            Category.ADMINISTRATIVE: ["schedule", "book", "arrange", "organize", "coordinate", "confirm"],
            Category.MEETING: ["meeting", "call", "conference", "discuss", "attend", "appointment"],
            Category.DEADLINE: ["due", "deadline", "submit", "deliver", "finish", "complete"],
            Category.PERSONAL: ["doctor", "dentist", "family", "friend", "home", "personal"]
        }

    def extract_tasks_from_transcript(self, transcript_text: str,
                                      speaker_info: Optional[Speaker] = None,
                                      meeting_context: Optional[str] = None) -> List[Task]:
        """Main method to extract tasks from meeting transcript."""
        logger.info(f"Starting task extraction from transcript ({len(transcript_text)} characters)")

        # Preprocess the text
        processed_text = self.nlp_processor.preprocess_text(transcript_text)

        # Resolve coreferences
        if self.config.get('task_extraction', {}).get('enable_coreference_resolution', True):
            processed_text = self.nlp_processor.resolve_coreferences(processed_text)

        # Extract sentences with metadata
        sentences = self.nlp_processor.extract_sentences(processed_text)

        # Extract tasks from sentences
        tasks = []
        for i, sentence_data in enumerate(sentences):
            sentence_tasks = self._extract_tasks_from_sentence(
                sentence_data,
                sentences[:i],  # Previous sentences for context
                speaker_info,
                meeting_context
            )
            tasks.extend(sentence_tasks)

        # Post-process tasks
        tasks = self._post_process_tasks(tasks)

        logger.info(f"Extracted {len(tasks)} tasks from transcript")
        return tasks

    def _extract_tasks_from_sentence(self, sentence_data: Dict,
                                     previous_sentences: List[Dict],
                                     speaker_info: Optional[Speaker],
                                     meeting_context: Optional[str]) -> List[Task]:
        """Extract tasks from a single sentence."""
        sentence_text = sentence_data['text']
        tasks = []

        # Check if sentence contains task indicators
        if not self._is_task_sentence(sentence_text):
            return tasks

        # Extract dates from sentence
        date_results = self.date_processor.extract_dates_from_text(sentence_text)

        # If no dates found, this might not be a task
        if not date_results:
            # Check for implicit tasks (meeting context might provide dates)
            if meeting_context and self._has_strong_task_indicators(sentence_text):
                date_results = [{'parsed_date': None, 'confidence': 0.5, 'method': 'implicit'}]
            else:
                return tasks

        # Create task for the best date match
        best_date_result = date_results[0] if date_results else None

        # Extract context clues
        context_clues = self.nlp_processor.extract_context_clues(
            sentence_text,
            [s['text'] for s in previous_sentences]
        )

        # Determine extraction method
        extraction_method = self._determine_extraction_method(sentence_text, context_clues)

        # Create task
        task = self._create_task_from_sentence(
            sentence_text,
            sentence_data,
            best_date_result,
            context_clues,
            extraction_method,
            speaker_info,
            meeting_context
        )

        if task:
            tasks.append(task)

        return tasks

    def _is_task_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a task."""
        sentence_lower = sentence.lower()

        # Check for action verbs
        has_action_verb = any(verb in sentence_lower for verb in self.action_verbs)

        # Check for task indicators
        has_task_indicator = any(indicator in sentence_lower for indicator in self.task_indicators)

        # Check for meeting-specific patterns
        has_meeting_pattern = any(
            re.search(pattern, sentence_lower) for pattern in self.meeting_specific_patterns
        )

        # Check for modal verbs indicating obligation/necessity
        has_modal_verb = bool(re.search(
            r'\b(should|must|need to|have to|ought to|will|shall|going to)\b',
            sentence_lower
        ))

        # Check for imperative sentences
        is_imperative = bool(re.search(
            r'^(please\s+)?(submit|complete|finish|send|deliver|review|prepare|create|update)',
            sentence.strip(),
            re.IGNORECASE
        ))

        return (has_action_verb or has_task_indicator or has_meeting_pattern or
                has_modal_verb or is_imperative)

    def _has_strong_task_indicators(self, sentence: str) -> bool:
        """Check for strong task indicators even without dates."""
        sentence_lower = sentence.lower()

        strong_indicators = [
            "action item", "to do", "todo", "assigned to", "responsible for",
            "will handle", "needs to be done", "follow up", "next steps"
        ]

        return any(indicator in sentence_lower for indicator in strong_indicators)

    def _determine_extraction_method(self, sentence: str, context_clues: Dict) -> ExtractionMethod:
        """Determine how the task was extracted."""
        sentence_lower = sentence.lower()

        # Check for pronouns that needed resolution
        if re.search(r'\b(this|that|it)\b', sentence_lower):
            if context_clues['subjects']:
                return ExtractionMethod.CONTEXTUAL
            else:
                return ExtractionMethod.INFERRED

        # Check if ML models were used (placeholder for actual ML classification)
        if self._used_ml_classification(sentence):
            return ExtractionMethod.ML_PREDICTED

        return ExtractionMethod.DIRECT

    def _used_ml_classification(self, sentence: str) -> bool:
        """Placeholder for ML classification check."""
        # In a real implementation, this would check if ML models were used
        return False

    def _create_task_from_sentence(self, sentence: str, sentence_data: Dict,
                                   date_result: Optional[Dict], context_clues: Dict,
                                   extraction_method: ExtractionMethod,
                                   speaker_info: Optional[Speaker],
                                   meeting_context: Optional[str]) -> Optional[Task]:
        """Create a Task object from sentence analysis."""

        # Clean the task text
        cleaned_text = self._clean_task_text(sentence, date_result)

        # Determine priority
        priority = self._determine_priority(sentence)

        # Determine category
        category = self._determine_category(sentence, sentence_data)

        # Calculate confidence
        confidence = self._calculate_task_confidence(
            sentence, date_result, context_clues, extraction_method
        )

        # Check if requires review
        requires_review = (
                confidence < self.config.get('task_extraction', {}).get('min_task_confidence', 0.6) or
                extraction_method in [ExtractionMethod.INFERRED, ExtractionMethod.CONTEXTUAL]
        )

        # Extract keywords
        keywords = self._extract_keywords(sentence, sentence_data)

        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            original_text=sentence,
            cleaned_text=cleaned_text,
            contextual_text=self._get_contextual_text(sentence,
                                                      context_clues) if extraction_method == ExtractionMethod.CONTEXTUAL else None,
            due_date=date_result['parsed_date'] if date_result else None,
            due_time=self.date_processor.extract_time_from_text(sentence).strftime('%H:%M') if date_result else None,
            priority=priority,
            category=category,
            status=TaskStatus.PENDING,
            confidence=confidence,
            extraction_method=extraction_method,
            requires_review=requires_review,
            speaker=speaker_info,
            meeting_context=meeting_context,
            keywords=keywords,
            mentioned_at=datetime.utcnow()
        )

        return task

    def _clean_task_text(self, sentence: str, date_result: Optional[Dict]) -> str:
        """Clean task text by removing date references and unnecessary words."""
        cleaned = sentence

        # Remove date references
        if date_result and date_result.get('original_text'):
            cleaned = cleaned.replace(date_result['original_text'], '').strip()

        # Remove common date patterns
        date_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(next|this)\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(by|before|after)\s+(end\s+of\s+)?(week|month|day)\b',
            r'\bin\s+\d+\s+(days?|weeks?|months?)\b'
        ]

        for pattern in date_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _determine_priority(self, sentence: str) -> Priority:
        """Determine task priority from sentence content."""
        sentence_lower = sentence.lower()

        for priority, keywords in self.priority_indicators.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return priority

        return Priority.MEDIUM

    def _determine_category(self, sentence: str, sentence_data: Dict) -> Category:
        """Determine task category from sentence content and metadata."""
        sentence_lower = sentence.lower()

        # Check entities for category clues
        entities = sentence_data.get('entities', [])
        entity_texts = [ent[0].lower() for ent in entities]

        for category, keywords in self.category_keywords.items():
            # Check in sentence text
            if any(keyword in sentence_lower for keyword in keywords):
                return category

            # Check in entities
            if any(keyword in entity_text for entity_text in entity_texts for keyword in keywords):
                return category

        return Category.WORK  # Default category

    def _calculate_task_confidence(self, sentence: str, date_result: Optional[Dict],
                                   context_clues: Dict, extraction_method: ExtractionMethod) -> float:
        """Calculate confidence score for task extraction."""
        confidence = 0.5  # Base confidence

        # Boost for clear action verbs
        if any(verb in sentence.lower() for verb in self.action_verbs):
            confidence += 0.2

        # Boost for task indicators
        if any(indicator in sentence.lower() for indicator in self.task_indicators):
            confidence += 0.2

        # Boost for date presence and quality
        if date_result:
            confidence += 0.1 + (date_result.get('confidence', 0) * 0.2)

        # Adjust based on extraction method
        method_adjustments = {
            ExtractionMethod.DIRECT: 0.1,
            ExtractionMethod.CONTEXTUAL: 0.0,
            ExtractionMethod.INFERRED: -0.1,
            ExtractionMethod.ML_PREDICTED: 0.05
        }
        confidence += method_adjustments.get(extraction_method, 0)

        # Boost for strong context clues
        if context_clues['subjects']:
            confidence += 0.1

        # Penalize very short or vague sentences
        if len(sentence.split()) < 4:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _extract_keywords(self, sentence: str, sentence_data: Dict) -> List[str]:
        """Extract relevant keywords from sentence."""
        keywords = []

        # Add action verbs found in sentence
        action_verbs_found = [verb for verb in self.action_verbs if verb in sentence.lower()]
        keywords.extend(action_verbs_found)

        # Add task indicators found
        indicators_found = [indicator for indicator in self.task_indicators if indicator in sentence.lower()]
        keywords.extend(indicators_found)

        # Add entities
        entities = sentence_data.get('entities', [])
        for entity_text, entity_label in entities:
            if entity_label in ['ORG', 'PERSON', 'PRODUCT', 'EVENT']:
                keywords.append(entity_text.lower())

        # Add important nouns
        pos_tags = sentence_data.get('pos_tags', [])
        important_nouns = [word.lower() for word, pos in pos_tags
                           if pos in ['NOUN', 'PROPN'] and len(word) > 3]
        keywords.extend(important_nouns[:3])  # Limit to top 3

        return list(set(keywords))  # Remove duplicates

    def _get_contextual_text(self, sentence: str, context_clues: Dict) -> Optional[str]:
        """Generate contextual text when references are resolved."""
        if not context_clues['subjects']:
            return None

        # Simple pronoun resolution
        contextual = sentence
        recent_subject = context_clues['subjects'][-1] if context_clues['subjects'] else None

        if recent_subject:
            contextual = re.sub(r'\bthis\b', recent_subject, contextual, flags=re.IGNORECASE)
            contextual = re.sub(r'\bthat\b', recent_subject, contextual, flags=re.IGNORECASE)
            contextual = re.sub(r'\bit\b', recent_subject, contextual, flags=re.IGNORECASE)

        return contextual if contextual != sentence else None

    def _post_process_tasks(self, tasks: List[Task]) -> List[Task]:
        """Post-process extracted tasks for quality and deduplication."""
        if not tasks:
            return tasks

        # Remove duplicates based on similarity
        unique_tasks = []
        for task in tasks:
            is_duplicate = False
            for existing_task in unique_tasks:
                similarity = self.nlp_processor.calculate_semantic_similarity(
                    task.cleaned_text, existing_task.cleaned_text
                )
                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    # Keep the task with higher confidence
                    if task.confidence > existing_task.confidence:
                        unique_tasks.remove(existing_task)
                        unique_tasks.append(task)
                    break

            if not is_duplicate:
                unique_tasks.append(task)

        # Sort by confidence and due date
        unique_tasks.sort(key=lambda t: (t.due_date or date.max, -t.confidence))

        return unique_tasks
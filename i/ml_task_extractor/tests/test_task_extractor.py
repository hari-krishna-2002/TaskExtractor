import pytest
import sys
from pathlib import Path
from datetime import date, datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.task_model import Task, Priority, Category, ExtractionMethod, Speaker
from utils.task_extractor import MLTaskExtractor
from utils.nlp_processor import NLPProcessor
from utils.date_processor import DateProcessor


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'models': {
            'spacy_model': 'en_core_web_sm',
            'transformer_model': 'microsoft/DialoGPT-medium',
            'sentence_transformer': 'all-MiniLM-L6-v2'
        },
        'task_extraction': {
            'min_task_confidence': 0.6,
            'enable_coreference_resolution': True,
            'enable_context_analysis': True
        },
        'date_parsing': {
            'prefer_future_dates': True,
            'timezone': 'UTC',
            'default_time': '09:00'
        }
    }


@pytest.fixture
def task_extractor(config):
    """Task extractor instance."""
    return MLTaskExtractor(config)


@pytest.fixture
def sample_speaker():
    """Sample speaker for testing."""
    return Speaker(name="John Doe", role="manager", authority_level=1.5)


class TestTaskExtractor:
    """Test cases for MLTaskExtractor."""

    def test_simple_task_extraction(self, task_extractor):
        """Test extraction of a simple task."""
        text = "We need to submit the quarterly report by tomorrow."
        tasks = task_extractor.extract_tasks_from_transcript(text)

        assert len(tasks) == 1
        task = tasks[0]
        assert "submit" in task.cleaned_text.lower()
        assert "quarterly report" in task.cleaned_text.lower()
        assert task.due_date is not None
        assert task.priority in [Priority.MEDIUM, Priority.HIGH]
        assert task.category == Category.WORK

    def test_multiple_tasks_extraction(self, task_extractor):
        """Test extraction of multiple tasks from text."""
        text = """
        We need to complete the project proposal by Friday. 
        Also, schedule a meeting with the client next week.
        Don't forget to review the budget tomorrow.
        """
        tasks = task_extractor.extract_tasks_from_transcript(text)

        assert len(tasks) >= 2  # Should extract at least 2 tasks

        # Check that different types of tasks are identified
        task_texts = [task.cleaned_text.lower() for task in tasks]
        assert any("proposal" in text for text in task_texts)
        assert any("meeting" in text for text in task_texts)

    def test_contextual_resolution(self, task_extractor):
        """Test contextual reference resolution."""
        text = """
        The quarterly financial report needs to be reviewed by the accounting team.
        This should be complete by tomorrow morning.
        """
        tasks = task_extractor.extract_tasks_from_transcript(text)

        # Should resolve "This" to refer to the report
        contextual_tasks = [task for task in tasks if task.extraction_method == ExtractionMethod.CONTEXTUAL]
        assert len(contextual_tasks) > 0

        contextual_task = contextual_tasks[0]
        assert contextual_task.contextual_text is not None
        assert "report" in contextual_task.contextual_text.lower()

    def test_priority_detection(self, task_extractor):
        """Test priority level detection."""
        test_cases = [
            ("This is urgent - submit the report immediately!", Priority.URGENT),
            ("We should complete this by next week", Priority.MEDIUM),
            ("Maybe we can consider this later", Priority.LOW),
            ("Critical deadline - finish by tomorrow!", Priority.HIGH)
        ]

        for text, expected_priority in test_cases:
            tasks = task_extractor.extract_tasks_from_transcript(text)
            if tasks:  # If a task was extracted
                assert tasks[0].priority == expected_priority

    def test_category_classification(self, task_extractor):
        """Test task category classification."""
        test_cases = [
            ("Schedule a client meeting for next week", Category.MEETING),
            ("Submit the project report by Friday", Category.WORK),
            ("Deploy the new system update", Category.TECHNICAL),
            ("Book a conference room for tomorrow", Category.ADMINISTRATIVE)
        ]

        for text, expected_category in test_cases:
            tasks = task_extractor.extract_tasks_from_transcript(text)
            if tasks:
                assert tasks[0].category == expected_category

    def test_speaker_context(self, task_extractor, sample_speaker):
        """Test speaker context in task extraction."""
        text = "I need everyone to complete their reports by Friday."
        tasks = task_extractor.extract_tasks_from_transcript(
            text,
            speaker_info=sample_speaker
        )

        assert len(tasks) > 0
        task = tasks[0]
        assert task.speaker is not None
        assert task.speaker.name == "John Doe"
        assert task.speaker.role == "manager"

    def test_meeting_context(self, task_extractor):
        """Test meeting context in task extraction."""
        text = "Let's make sure the presentation is ready for the board meeting."
        meeting_context = "Q4 Planning Meeting"

        tasks = task_extractor.extract_tasks_from_transcript(
            text,
            meeting_context=meeting_context
        )

        assert len(tasks) > 0
        task = tasks[0]
        assert task.meeting_context == meeting_context

    def test_confidence_scoring(self, task_extractor):
        """Test confidence scoring for extracted tasks."""
        # High confidence task (clear action, date, subject)
        high_conf_text = "Submit the quarterly report by tomorrow at 5 PM."
        high_conf_tasks = task_extractor.extract_tasks_from_transcript(high_conf_text)

        # Low confidence task (vague, no clear date)
        low_conf_text = "Maybe we should think about that thing sometime."
        low_conf_tasks = task_extractor.extract_tasks_from_transcript(low_conf_text)

        if high_conf_tasks:
            assert high_conf_tasks[0].confidence > 0.7

        if low_conf_tasks:
            assert low_conf_tasks[0].confidence < 0.5

    def test_date_extraction_accuracy(self, task_extractor):
        """Test accuracy of date extraction."""
        today = date.today()

        test_cases = [
            ("Complete this by tomorrow", 1),  # 1 day from today
            ("Finish by next Friday", None),  # Variable, but should be future
            ("Due today", 0),  # Today
        ]

        for text, expected_days_offset in test_cases:
            tasks = task_extractor.extract_tasks_from_transcript(text)
            if tasks and tasks[0].due_date:
                if expected_days_offset is not None:
                    expected_date = today + timedelta(days=expected_days_offset)
                    assert tasks[0].due_date == expected_date
                else:
                    # Should be a future date
                    assert tasks[0].due_date > today

    def test_keyword_extraction(self, task_extractor):
        """Test keyword extraction from tasks."""
        text = "We need to urgently review and submit the client proposal by Friday."
        tasks = task_extractor.extract_tasks_from_transcript(text)

        assert len(tasks) > 0
        task = tasks[0]

        # Should extract relevant keywords
        keywords = [kw.lower() for kw in task.keywords]
        assert any(kw in keywords for kw in ['review', 'submit', 'proposal', 'client'])

    def test_duplicate_removal(self, task_extractor):
        """Test removal of duplicate tasks."""
        text = """
        Submit the report by tomorrow.
        The report needs to be submitted by tomorrow.
        We should submit the report tomorrow.
        """
        tasks = task_extractor.extract_tasks_from_transcript(text)

        # Should not have 3 identical tasks
        assert len(tasks) < 3

        # Remaining tasks should have reasonable confidence
        for task in tasks:
            assert task.confidence > 0.5


class TestDateProcessor:
    """Test cases for DateProcessor."""

    def test_relative_date_parsing(self):
        """Test parsing of relative dates."""
        config = {'date_parsing': {'prefer_future_dates': True, 'timezone': 'UTC'}}
        processor = DateProcessor(config)

        test_cases = [
            "Complete this by tomorrow",
            "Due next Friday",
            "Finish in 3 days",
            "Submit by end of week"
        ]

        for text in test_cases:
            results = processor.extract_dates_from_text(text)
            assert len(results) > 0
            assert results[0]['parsed_date'] is not None
            assert results[0]['confidence'] > 0.5

    def test_urgency_detection(self):
        """Test urgency level detection."""
        config = {'date_parsing': {'prefer_future_dates': True, 'timezone': 'UTC'}}
        processor = DateProcessor(config)

        test_cases = [
            ("This is urgent!", "urgent"),
            ("Please do this soon", "high"),
            ("Complete by end of week", "medium"),
            ("When you have time", "low")
        ]

        for text, expected_urgency in test_cases:
            urgency, confidence = processor.determine_urgency(text)
            assert urgency == expected_urgency
            assert confidence > 0.5


class TestNLPProcessor:
    """Test cases for NLPProcessor."""

    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        processor = NLPProcessor()

        raw_text = "Um, we need to, uh, complete the report, you know?"
        processed = processor.preprocess_text(raw_text)

        # Should remove filler words
        assert "um" not in processed.lower()
        assert "uh" not in processed.lower()
        assert "you know" not in processed.lower()

    def test_entity_extraction(self):
        """Test named entity extraction."""
        processor = NLPProcessor()

        text = "John Smith from Microsoft needs to review the Q4 report by December 15th."
        entities = processor.extract_entities(text)

        # Should extract person, organization, and date entities
        assert 'PERSON' in entities or 'ORG' in entities
        assert len(entities) > 0

    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        processor = NLPProcessor()

        text1 = "Submit the quarterly report"
        text2 = "Send in the Q4 report"
        text3 = "Buy groceries"

        # Similar texts should have high similarity
        similarity_high = processor.calculate_semantic_similarity(text1, text2)
        similarity_low = processor.calculate_semantic_similarity(text1, text3)

        assert similarity_high > similarity_low
        assert similarity_high > 0.5
        assert similarity_low < 0.5


if __name__ == "__main__":
    pytest.main([__file__])
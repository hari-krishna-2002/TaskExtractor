import dateparser
from datetime import datetime, date, time, timedelta
from dateparser.search import search_dates
import re
from typing import Optional, List, Dict, Tuple
from loguru import logger
import pytz


class DateProcessor:
    def __init__(self, config: Dict):
        self.config = config.get('date_parsing', {})
        self.timezone = pytz.timezone(self.config.get('timezone', 'UTC'))
        self.default_time = self.config.get('default_time', '09:00')
        self.prefer_future = self.config.get('prefer_future_dates', True)

        # Enhanced date patterns for meeting contexts
        self.date_patterns = [
            # Explicit dates
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',

            # Relative dates
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(next|this)\s+(week|month|year)\b',
            r'\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',

            # Time-based
            r'\b(in|within)\s+(\d+)\s+(days?|weeks?|months?|hours?|minutes?)\b',
            r'\b(by|before|after)\s+(end\s+of\s+)?(week|month|day|today|tomorrow)\b',

            # Meeting-specific patterns
            r'\b(by\s+the\s+end\s+of\s+the\s+meeting)\b',
            r'\b(before\s+we\s+meet\s+again)\b',
            r'\b(by\s+our\s+next\s+meeting)\b',
            r'\b(end\s+of\s+business\s+day)\b',
            r'\b(eod|eob|asap)\b',

            # Time patterns
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
        ]

        # Urgency indicators
        self.urgency_patterns = {
            'urgent': r'\b(urgent|asap|immediately|right away|as soon as possible)\b',
            'high': r'\b(soon|quickly|priority|important|critical)\b',
            'medium': r'\b(by\s+end\s+of\s+week|this\s+week|next\s+week)\b',
            'low': r'\b(when\s+possible|eventually|sometime|later)\b'
        }

    def extract_dates_from_text(self, text: str) -> List[Dict]:
        """Extract all date references from text with confidence scores."""
        results = []

        # Use dateparser's search_dates function
        try:
            search_results = search_dates(
                text,
                settings={
                    'PREFER_DATES_FROM': 'future' if self.prefer_future else 'current_period',
                    'TIMEZONE': str(self.timezone),
                    'RETURN_AS_TIMEZONE_AWARE': True
                }
            )

            if search_results:
                for date_string, parsed_date in search_results:
                    confidence = self._calculate_date_confidence(date_string, text)
                    results.append({
                        'original_text': date_string,
                        'parsed_date': parsed_date.date() if parsed_date else None,
                        'parsed_time': parsed_date.time() if parsed_date else None,
                        'confidence': confidence,
                        'method': 'dateparser'
                    })

        except Exception as e:
            logger.warning(f"Dateparser failed: {e}")

        # Fallback to pattern matching
        pattern_results = self._extract_dates_by_patterns(text)
        results.extend(pattern_results)

        # Remove duplicates and sort by confidence
        results = self._deduplicate_dates(results)
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def _extract_dates_by_patterns(self, text: str) -> List[Dict]:
        """Extract dates using regex patterns."""
        results = []

        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group()
                parsed_date = self._parse_relative_date(date_text)

                if parsed_date:
                    confidence = self._calculate_pattern_confidence(date_text, pattern)
                    results.append({
                        'original_text': date_text,
                        'parsed_date': parsed_date,
                        'parsed_time': None,
                        'confidence': confidence,
                        'method': 'pattern_matching'
                    })

        return results

    def _parse_relative_date(self, date_text: str) -> Optional[date]:
        """Parse relative date expressions."""
        today = datetime.now(self.timezone).date()
        date_text = date_text.lower().strip()

        # Simple relative dates
        if date_text == 'today':
            return today
        elif date_text == 'tomorrow':
            return today + timedelta(days=1)
        elif date_text == 'yesterday':
            return today - timedelta(days=1)

        # Next/this week patterns
        if 'next week' in date_text:
            days_ahead = 7 - today.weekday() + 7
            return today + timedelta(days=days_ahead)
        elif 'this week' in date_text:
            days_ahead = 4 - today.weekday()  # Friday
            return today + timedelta(days=days_ahead)

        # Weekday patterns
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }

        for day_name, day_num in weekdays.items():
            if day_name in date_text:
                days_ahead = day_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                if 'next' in date_text:
                    days_ahead += 7
                return today + timedelta(days=days_ahead)

        # In X days pattern
        in_days_match = re.search(r'in\s+(\d+)\s+days?', date_text)
        if in_days_match:
            days = int(in_days_match.group(1))
            return today + timedelta(days=days)

        return None

    def _calculate_date_confidence(self, date_string: str, full_text: str) -> float:
        """Calculate confidence score for date extraction."""
        confidence = 0.5  # Base confidence

        # Boost confidence for explicit dates
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_string):
            confidence += 0.3

        # Boost for common relative dates
        if date_string.lower() in ['today', 'tomorrow', 'yesterday']:
            confidence += 0.4

        # Boost for context clues
        context_words = ['due', 'deadline', 'by', 'before', 'until', 'complete']
        for word in context_words:
            if word in full_text.lower():
                confidence += 0.1
                break

        # Reduce confidence for ambiguous patterns
        if len(date_string) < 4:
            confidence -= 0.2

        return min(1.0, max(0.0, confidence))

    def _calculate_pattern_confidence(self, date_text: str, pattern: str) -> float:
        """Calculate confidence for pattern-matched dates."""
        # This is a simplified confidence calculation
        # In production, you'd have more sophisticated scoring
        base_confidence = 0.6

        if 'today' in date_text.lower() or 'tomorrow' in date_text.lower():
            return 0.9
        elif any(day in date_text.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            return 0.8
        elif re.search(r'\d', date_text):
            return 0.7

        return base_confidence

    def _deduplicate_dates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate date extractions."""
        seen_dates = set()
        unique_results = []

        for result in results:
            if result['parsed_date']:
                date_key = result['parsed_date'].isoformat()
                if date_key not in seen_dates:
                    seen_dates.add(date_key)
                    unique_results.append(result)

        return unique_results

    def determine_urgency(self, text: str) -> Tuple[str, float]:
        """Determine urgency level from text."""
        text_lower = text.lower()

        for urgency_level, pattern in self.urgency_patterns.items():
            if re.search(pattern, text_lower):
                confidence = 0.8 if urgency_level in ['urgent', 'high'] else 0.6
                return urgency_level, confidence

        return 'medium', 0.5

    def extract_time_from_text(self, text: str) -> Optional[time]:
        """Extract time information from text."""
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3:  # HH:MM am/pm
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        period = match.group(3)

                        if period and period.lower() == 'pm' and hour != 12:
                            hour += 12
                        elif period and period.lower() == 'am' and hour == 12:
                            hour = 0

                        return time(hour, minute)

                    elif len(match.groups()) == 2:  # H am/pm
                        hour = int(match.group(1))
                        period = match.group(2)

                        if period and period.lower() == 'pm' and hour != 12:
                            hour += 12
                        elif period and period.lower() == 'am' and hour == 12:
                            hour = 0

                        return time(hour, 0)

                except ValueError:
                    continue

        # Default time if no specific time found
        default_hour, default_minute = map(int, self.default_time.split(':'))
        return time(default_hour, default_minute)
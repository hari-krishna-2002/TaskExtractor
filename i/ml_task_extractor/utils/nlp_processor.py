import spacy
import re
from typing import List, Dict, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
import yaml
from pathlib import Path


class NLPProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # def _load_config(self, config_path: str) -> Dict:
    #     """Load configuration from YAML file."""
    #     config_file = Path(__file__).parent.parent / config_path
    #     with open(config_file, 'r') as f:
    #         return yaml.safe_load(f)

    def _initialize_models(self):
        """Initialize all NLP models."""
        try:
            # SpaCy model for basic NLP tasks
            self.nlp = spacy.load(self.config['models']['spacy_model'])

            # Add custom components
            if 'coref' not in self.nlp.pipe_names:
                # Note: In production, you'd add neuralcoref or similar
                logger.info("Coreference resolution not available, using rule-based approach")

            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer(
                self.config['models']['sentence_transformer']
            )

            # Task classification model (you can train a custom one)
            self.task_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )

            logger.info("All NLP models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove filler words common in speech
        filler_words = r'\b(um|uh|er|ah|like|you know|sort of|kind of)\b'
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)

        # Fix common speech-to-text errors
        text = re.sub(r'\bwanna\b', 'want to', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgonna\b', 'going to', text, flags=re.IGNORECASE)

        return text.strip()

    def extract_sentences(self, text: str) -> List[Dict]:
        """Extract sentences with metadata."""
        doc = self.nlp(self.preprocess_text(text))
        sentences = []

        for sent in doc.sents:
            sentence_data = {
                'text': sent.text.strip(),
                'start': sent.start_char,
                'end': sent.end_char,
                'tokens': [token.text for token in sent],
                'pos_tags': [(token.text, token.pos_) for token in sent],
                'entities': [(ent.text, ent.label_) for ent in sent.ents],
                'dependencies': [(token.text, token.dep_, token.head.text) for token in sent]
            }
            sentences.append(sentence_data)

        return sentences

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        return entities

    def resolve_coreferences(self, text: str) -> str:
        """Resolve coreferences in text (rule-based approach)."""
        # This is a simplified rule-based approach
        # In production, use neuralcoref or similar

        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Simple pronoun resolution
        resolved_sentences = []
        recent_subjects = []

        for sentence in sentences:
            sent_doc = self.nlp(sentence)

            # Extract subjects from current sentence
            subjects = []
            for token in sent_doc:
                if token.dep_ == "nsubj" and token.pos_ == "NOUN":
                    subjects.append(token.text)

            recent_subjects.extend(subjects)
            recent_subjects = recent_subjects[-3:]  # Keep last 3 subjects

            # Replace pronouns
            resolved = sentence
            if recent_subjects:
                resolved = re.sub(r'\bthis\b', recent_subjects[-1], resolved, flags=re.IGNORECASE)
                resolved = re.sub(r'\bthat\b', recent_subjects[-1], resolved, flags=re.IGNORECASE)
                if len(recent_subjects) > 0:
                    resolved = re.sub(r'\bit\b', recent_subjects[-1], resolved, flags=re.IGNORECASE)

            resolved_sentences.append(resolved)

        return ' '.join(resolved_sentences)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs from text."""
        doc = self.nlp(text)
        action_verbs = []

        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp", "ccomp"]:
                action_verbs.append(token.lemma_)

        return action_verbs

    def identify_temporal_expressions(self, text: str) -> List[Dict]:
        """Identify temporal expressions in text."""
        doc = self.nlp(text)
        temporal_expressions = []

        # Look for DATE entities
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                temporal_expressions.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Look for temporal patterns
        temporal_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(next|this)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(in|within)\s+\d+\s+(days?|weeks?|months?)\b',
            r'\b(by|before|after)\s+\w+\b',
            r'\b\d{1,2}[:/]\d{1,2}(?:[:/]\d{1,2})?\b',  # Time patterns
        ]

        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_expressions.append({
                    'text': match.group(),
                    'label': 'TEMPORAL_PATTERN',
                    'start': match.start(),
                    'end': match.end()
                })

        return temporal_expressions

    def extract_context_clues(self, sentence: str, previous_sentences: List[str]) -> Dict:
        """Extract context clues from surrounding sentences."""
        context_clues = {
            'subjects': [],
            'topics': [],
            'entities': {},
            'semantic_similarity': []
        }

        # Analyze previous sentences for context
        for prev_sentence in previous_sentences[-3:]:  # Last 3 sentences
            prev_doc = self.nlp(prev_sentence)

            # Extract subjects
            for token in prev_doc:
                if token.dep_ == "nsubj":
                    context_clues['subjects'].append(token.text)

            # Extract entities
            for ent in prev_doc.ents:
                if ent.label_ not in context_clues['entities']:
                    context_clues['entities'][ent.label_] = []
                context_clues['entities'][ent.label_].append(ent.text)

            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(sentence, prev_sentence)
            context_clues['semantic_similarity'].append(similarity)

        return context_clues
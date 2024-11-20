import json
from typing import Optional
from pathlib import Path

class FAQBot:
    def __init__(self, faq_path: str):
        """Initialize FAQBot with FAQ data"""
        self.faq_data = self._load_faq_data(faq_path)
        self.question_map = {
            q['question'].lower(): q['answer'] 
            for q in self.faq_data
        }

    def _load_faq_data(self, faq_path: str) -> list:
        """Load FAQ data from JSON file"""
        path = Path(faq_path)
        if not path.exists():
            raise FileNotFoundError(f"FAQ data file not found: {faq_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_exact_match(self, question: str) -> Optional[str]:
        """Find exact match for question in FAQ data"""
        return self.question_map.get(question.lower())

    async def get_response(self, question: str, chat_model=None) -> str:
        """
        Get response for a question using following priority:
        1. Exact match from FAQ
        2. Chat model response (if provided)
        3. Default "not found" response
        """
        # Try exact match first
        exact_match = self.find_exact_match(question)
        if exact_match:
            return exact_match

        # Try chat model if provided
        if chat_model:
            try:
                response = await chat_model.agenerate_response(question)
                return response
            except Exception as e:
                print(f"Error using chat model: {e}")

        # Default response if no match found
        return "I'm sorry, I couldn't find an answer for that question." 
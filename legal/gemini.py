import os
import time
import google.generativeai as genai
from typing import Optional

class GeminiChat:
    def __init__(self, api_key: str, default_model: str = "gemini-pro", max_retries: int = 3):
        self.api_key = api_key
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = 2
        self.last_prompt = None
        self.last_response = None
        self._configure_genai()

    def _configure_genai(self):
        os.environ["GOOGLE_API_KEY"] = self.api_key
        genai.configure(api_key=self.api_key)

    def _rate_limit(self):
        time.sleep(1)  # Basic rate-limiting to avoid overloading the API

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        if context:
            return f"Context: {context}\n\nQuestion: {prompt}"
        return prompt

    def _format_response(self, text: str) -> str:
        return text.strip()

    def _summarize_response(self, text: str) -> str:
        summary_lines = text.split("\n")[:5]
        return "\n".join(summary_lines) + "\n\n📌 Ask a follow-up for more details."

    def generate_response(self, prompt: str, context: Optional[str] = None, temperature: float = 0.3, model_name: Optional[str] = None) -> str:
        self._rate_limit()

        if prompt.strip().lower() == (self.last_prompt or "").strip().lower():
            return f"🔁 You've already asked this. Here's a brief recap:\n\n{self._summarize_response(self.last_response)}"

        model_to_use = model_name or self.default_model
        full_prompt = self._build_prompt(prompt, context)

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(model_to_use)
                response = model.generate_content(full_prompt, generation_config={"temperature": temperature})
                formatted_response = self._format_response(response.text)

                self.last_prompt = prompt
                self.last_response = formatted_response

                return formatted_response

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"❌ Failed to generate response after {self.max_retries} attempts. Error: {str(e)}"

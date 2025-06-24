import google.generativeai as genai
import time
from typing import Optional

class GeminiChat:
    def __init__(self):
        self.last_request_time = 0
        self.request_delay = 1.2  # seconds between requests
        self.current_model = "gemini-1.5-flash"  # Updated to Flash 2.0 equivalent
        self.fallback_model = "gemini-1.0-pro"
        self.max_retries = 2
        
        # System prompt optimized for Indian legal context
        self.system_prompt = """
        You are VaakeelAI, an expert Indian legal assistant specializing in:
        - Constitutional Law (Articles 14-32, 226, 370)
        - Civil Procedure Code
        - Criminal Procedure Code
        - IPC, Evidence Act
        - Consumer Protection Act
        - Property and Tenancy Laws
        
        Response Guidelines:
        1. Cite relevant sections (e.g., "Under IPC Section 420...")
        2. Explain like you're teaching law students
        3. Use Indian legal terminology
        4. Flag when professional consultation is advised
        5. Never predict case outcomes
        6. For documents: summarize key clauses, flag anomalies
        """

    def _rate_limit(self):
        """Enforce delay between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """Generate legal response with automatic fallback handling"""
        self._rate_limit()
        
        generation_config = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 32,
            "max_output_tokens": 4096,  # Reduced for cost efficiency
        }
        
        model = genai.GenerativeModel(self.current_model)
        full_prompt = self._build_prompt(prompt, context)
        
        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    safety_settings={
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )
                return self._format_response(response.text)
                
            except Exception as e:
                if "quota" in str(e).lower() and self.current_model != self.fallback_model:
                    self.current_model = self.fallback_model
                    model = genai.GenerativeModel(self.current_model)
                    continue
                return self._handle_error(e)
        
        return "⚠️ System busy. Please try again in a moment."

    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        """Construct the full prompt with context"""
        if context:
            return f"""
            {self.system_prompt}
            
            [DOCUMENT CONTEXT]
            {context[:6000]}  # Truncate to save tokens
            
            [USER QUESTION]
            {prompt}
            
            Analyze the documents above and provide:
            1. Relevant legal provisions
            2. Potential issues flagged
            3. Next steps suggested
            """
        return f"""
        {self.system_prompt}
        
        [USER QUESTION]
        {prompt}
        
        Provide detailed response covering:
        1. Applicable laws/sections
        2. Judicial precedents if available
        3. Practical considerations
        """

    def _format_response(self, text: str) -> str:
        """Format the raw API response"""
        # Add section headers for readability
        replacements = {
            "IPC Section": "📜 **IPC Section**",
            "Article": "⚖️ **Article**",
            "Case Law": "\n\n**Relevant Case Law:**\n",
            "Advice": "\n\n❗ **Important Note:**"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _handle_error(self, error: Exception) -> str:
        """Generate user-friendly error messages"""
        error_msg = str(error)
        
        if "quota" in error_msg.lower():
            return ("⚠️ Our systems are busy (quota limit reached). "
                   "Please try:\n"
                   "1. Simplifying your question\n"
                   "2. Waiting 1-2 minutes\n"
                   "3. Uploading smaller documents")
        
        elif "safety" in error_msg.lower():
            return ("🔒 Response blocked by safety filters. "
                   "Please rephrase your question avoiding "
                   "personal/identifiable information.")
        
        return f"⚠️ Technical difficulty: {error_msg[:200]}... Please try again."


# Singleton instance for the app
legal_assistant = GeminiChat()

def gemini_chat(prompt: str, context: Optional[str] = None) -> str:
    """Public interface for chat functionality"""
    return legal_assistant.generate_response(prompt, context)

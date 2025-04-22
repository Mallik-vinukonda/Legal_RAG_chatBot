# Gemini chat logic will go here.

import google.generativeai as genai

def gemini_chat(prompt, context=None, temperature=0.2):
    """Generate response using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        system_prompt = """
        You are VadheelAI, an expert Indian legal assistant with expertise in Indian constitutional law, civil law, criminal law, 
        and other legal domains relevant to India. Your task is to provide accurate, clear, and helpful responses to legal queries.
        Guidelines:
        1. Always cite relevant sections of laws, acts, or constitutional articles when applicable
        2. Explain legal concepts in simple, understandable language
        3. When the context is unclear, ask clarifying questions
        4. Avoid providing definitive legal advice; instead offer information and general guidance
        5. Make it clear when a question requires specialized legal expertise beyond general knowledge
        6. Be respectful of the Indian legal system and its processes
        When analyzing documents, explain their content clearly and identify any potential legal issues.
        """
        if context:
            full_prompt = f"""
            {system_prompt}
            CONTEXT INFORMATION:
            {context}
            USER QUERY:
            {prompt}
            Please provide a detailed, helpful response based on the context and your knowledge of Indian law.
            """
        else:
            full_prompt = f"""
            {system_prompt}
            USER QUERY:
            {prompt}
            Please provide a detailed, helpful response based on your knowledge of Indian law.
            """
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

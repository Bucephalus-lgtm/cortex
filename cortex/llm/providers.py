from cortex.base import LLMProvider
from typing import Optional
import os

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.api_key:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            return "OpenAI API key not configured. (Mock response: Use LLM for better answers)"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

class OllamaProvider(LLMProvider):
    """Local LLM provider using Ollama. Optimized for Mac (Metal) and lightweight models."""
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=30
            )
            return response.json().get("response", "Error: No response from Ollama")
        except Exception as e:
            return f"Ollama Error: {str(e)}. (Hint: Is Ollama running and '{self.model}' pulled?)"

class GroqProvider(LLMProvider):
    """Cloud provider using Groq for lightning-fast inference with ZERO local PC impact."""
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-8b-8192"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if self.api_key:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            return "Groq API key not configured. (Zero impact, lightning fast, but needs simple API key)"
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "This is a mock LLM response. In a real setup, I would process the context and question to provide a grounded answer."

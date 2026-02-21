import os
from contextlib import contextmanager

# Simple token approx (1 token ~ 4 characters)
def count_tokens(text: str) -> int:
    return len(text) // 4

def truncate_context(context_chunks: list[str], max_tokens: int = 3000) -> str:
    """
    Context window management:
    Takes retrieved chunks and fits them into the prompt's context window.
    """
    selected_chunks = []
    current_tokens = 0
    for chunk in context_chunks:
        tokens = count_tokens(chunk)
        if current_tokens + tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        current_tokens += tokens
    return "\n\n---\n\n".join(selected_chunks)

class LLMProvider:
    def generate_answer(self, question: str, context: str) -> str:
        raise NotImplementedError

class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None

    def generate_answer(self, question: str, context: str) -> str:
        if not self.client:
            return "OpenAI API client not configured or 'openai' package missing. Please set OPENAI_API_KEY."
            
        system_prompt = (
            "You are an expert Production Incident Copilot. "
            "You must answer the user's question based strictly on the provided context. "
            "If the answer is not in the context, reply exactly with 'Answer not found in knowledge base.' "
            "Do not hallucinate external information."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content

class MockLLM(LLMProvider):
    def generate_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "Answer not found in knowledge base."
        return f"Mocked LLM Answer based on ({len(context)} chars of context):\n" + context[:200] + "..."

def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    # Fallback to Mock or Local HuggingFace as needed
    return MockLLM()

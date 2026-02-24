from typing import List, Dict, Any, Optional
from cortex.base import Retriever, LLMProvider
from cortex.utils.prompt import PromptManager
from cortex.utils.context import ContextManager

class CortexEngine:
    def __init__(self, retriever: Retriever, llm_provider: LLMProvider, context_window: int = 2048):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.context_manager = ContextManager(max_tokens=context_window)
        self.prompt_manager = PromptManager()

    def ask(self, question: str) -> Dict[str, Any]:
        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=5)
        
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "Answer not found in knowledge base.",
                "sources": []
            }

        # 2. Prepare context
        context = self.context_manager.prepare_context(retrieved_chunks)
        
        # 3. Generate answer using LLM
        system_prompt = self.prompt_manager.get_system_prompt()
        user_prompt = self.prompt_manager.format_prompt(question, context)
        
        answer = self.llm_provider.generate(user_prompt, system_prompt=system_prompt)
        
        # 4. Extract sources
        sources = list({c.get("source") for c in retrieved_chunks if c.get("source")})
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

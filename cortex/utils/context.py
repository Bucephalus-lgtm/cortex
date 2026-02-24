from typing import List, Dict, Any

class ContextManager:
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens

    def prepare_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        # Simple character-based estimation (approx 4 chars per token)
        max_chars = self.max_tokens * 4
        current_chars = 0
        context_parts = []

        for chunk in retrieved_chunks:
            text = chunk.get("text", "")
            source = chunk.get("source", "Unknown")
            formatted_chunk = f"[Source: {source}]\n{text}\n"
            
            if current_chars + len(formatted_chunk) > max_chars:
                # Could trim the chunk here if we wanted to be more aggressive
                break
            
            context_parts.append(formatted_chunk)
            current_chars += len(formatted_chunk)
        
        return "\n".join(context_parts)

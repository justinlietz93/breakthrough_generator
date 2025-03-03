"""
ai_clients.py

Minimal approach to calling:
1) Claude 3.7 Sonnet (Anthropic-based)
2) DeepSeek R1 (OpenAI-based approach)

No streaming, no chunker, just a single .run(...) method that returns final text.
"""

import os
import anthropic
import openai

class Claude37SonnetClient:
    """
    Minimal client for Claude 3.7 Sonnet. 
    Uses environment variables:
      - ANTHROPIC_API_KEY: The key for Anthropic
      - CLAUDE_MODEL (optional, default "claude-3-7-sonnet-20250219")
    """

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "missing-api-key")
        self.model_name = os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def run(self, messages, max_tokens=4096, temperature=0.0, enable_thinking=False, thinking_budget=None):
        """
        Non-stream call to the Claude 3.7 Sonnet model.
        :param messages: list of { "role": "user"/"assistant"/"system", "content": "..."}
        :param max_tokens: limit for the generated text
        :param temperature: Controls randomness (0.0 to 1.0)
        :param enable_thinking: Whether to enable Claude's extended thinking capability
        :param thinking_budget: Number of tokens for thinking (min 1024, default to max_tokens - 1000)
        :return: final string
        """
        try:
            # Extract system message if present
            system_prompt = None
            filtered_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    filtered_messages.append(msg)
            
            # Create the request
            params = {
                "model": self.model_name,
                "messages": filtered_messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add system prompt if present
            if system_prompt:
                params["system"] = system_prompt
                
            # Add thinking if enabled
            if enable_thinking:
                # Default thinking budget is max_tokens minus a buffer, or 1024 if that would be too small
                if thinking_budget is None:
                    thinking_budget = max(1024, max_tokens - 1000)
                
                # Ensure minimum of 1024 tokens and less than max_tokens
                thinking_budget = max(1024, min(thinking_budget, max_tokens - 100))
                
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
                
            resp = self.client.messages.create(**params)
            
            # Extract content from response
            result = ""
            if resp.content and len(resp.content) > 0:
                # Look for thinking content blocks first
                has_thinking = False
                thinking_text = ""
                answer_text = ""
                
                for block in resp.content:
                    if hasattr(block, 'type'):
                        if block.type == "thinking":
                            has_thinking = True
                            thinking_text = block.thinking
                        elif block.type == "text":
                            answer_text += block.text
                
                # Format the response with thinking if available
                if has_thinking:
                    result = f"Thinking:\n{thinking_text}\n\nAnswer:\n{answer_text}"
                else:
                    result = answer_text or resp.content[0].text
                
                return result
            return ""
        except Exception as e:
            return f"ERROR from Claude: {str(e)}"


class DeepseekR1Client:
    """
    Minimal client for DeepSeek R1 using openai library with a custom base URL.
    Env variables:
      - DEEPSEEK_API_KEY
    """

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "missing-deepseek-key")
        # Create a proper client instance using the modern SDK pattern
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model_name = "deepseek-reasoner"

    def run(self, messages, max_tokens=8000, temperature=0.0):
        """
        Non-stream call to DeepSeek R1
        
        :param messages: list of { "role": "user"/"assistant"/"system", "content": "..."}
        :param max_tokens: limit for the generated text
        :param temperature: Controls randomness (0.0 to 2.0, lower is better for coding)
        :return: final string including reasoning if available
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            if resp.choices and len(resp.choices) > 0:
                # Check if reasoning content is available (deepseek-reasoner specific)
                reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
                content = resp.choices[0].message.content
                
                # If reasoning is available, prepend it to the content
                if reasoning:
                    return f"Reasoning:\n{reasoning}\n\nAnswer:\n{content}"
                return content
            return ""
        except Exception as e:
            return f"ERROR from DeepSeek: {str(e)}"


class AIOrchestrator:
    """
    A minimal orchestrator that picks either Claude3.7Sonnet or DeepseekR1
    and calls .run(...) with system+user messages.
    """

    def __init__(self, model_name: str):
        """
        model_name can be "claude37sonnet" or "deepseekr1"
        """
        self.model_name = model_name.lower()
        if self.model_name == "claude37sonnet":
            self.client = Claude37SonnetClient()
        elif self.model_name == "deepseekr1":
            self.client = DeepseekR1Client()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
        """
        Minimal synergy: just pass system+user messages, get final text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.client.run(messages, max_tokens=max_tokens)

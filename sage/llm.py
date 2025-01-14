import os
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

def build_llm_via_langchain(provider: str, model: str):
    """Builds a language model via LangChain."""
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        return ChatOpenAI(model=model or "gpt-4")
    elif provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
        return ChatAnthropic(model=model or "claude-3-opus-20240229")
    elif provider == "ollama":
        return ChatOllama(model=model or "llama3.1")
    elif provider == "together":
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable.")
        return ChatTogether(model=model or "deepseek-ai/DeepSeek-V3")
    else:
        raise ValueError(f"Unrecognized LLM provider {provider}. Contributons are welcome!")

from typing import Optional
from class_registry import ClassRegistry

from GDesigner.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None, use_cache: bool = False) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o"

        # Check if model_name is directly registered (e.g., 'hybrid_cache', 'qwen-plus')
        if model_name in cls.registry.keys():
            model = cls.registry.get(model_name)
        elif model_name == 'mock':
            model = cls.registry.get(model_name)
        elif use_cache:
            # Fallback: try to use cache-enabled wrapper (legacy)
            model = cls.registry.get('GPTChat', model_name)
        else:
            # Standard LLM
            model = cls.registry.get('GPTChat', model_name)

        return model

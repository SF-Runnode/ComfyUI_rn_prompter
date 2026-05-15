from .prompter import RN_Prompt_Translator, RN_Midjourney_Prompter, RN_LLMAPI_Node, RN_LLMAPI_Pro_Node, RN_Translator

NODE_CLASS_MAPPINGS = {
    "RN Translator": RN_Translator,
    "RN Prompt Translator": RN_Prompt_Translator,
    "RN Midjourney Prompter": RN_Midjourney_Prompter,
    "RN LLM API": RN_LLMAPI_Node,
    "RN LLM API Pro": RN_LLMAPI_Pro_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN Translator": "RunNode Translator",
    "RN Prompt Translator": "RunNode Prompt Translator",
    "RN Midjourney Prompter": "Midjourney Style Prompter",
    "RN LLM API": "RunNode LLM API Node",
    "RN LLM API Pro": "RunNode LLM API Pro Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "1.2"

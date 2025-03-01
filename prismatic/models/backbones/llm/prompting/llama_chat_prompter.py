"""
llama2_prompter.py

Defines a PromptBuilder for building Llama-2 Chat Prompts --> not sure if this is "optimal", but this is the pattern
that's used by HF and other online tutorials.

Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for Prismatic Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
}


def format_system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class Llama2ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # Llama-2 Specific
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"[INST] {msg} [/INST] "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()



def format_system_prompt3(system_prompt: str) -> str:
    """Format the system prompt by stripping whitespace and adding newlines."""
    return f"{system_prompt.strip()}\n"



class Llama3ChatPromptBuilder(PromptBuilder):
    """
    The Llama3 format follows the pattern:
    <|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {assistant_message}<|eot_id|>

    Attributes:
        system_prompt (str): The system prompt for the conversation.
        prompt (str): The accumulated conversation history.
        turn_count (int): Counter tracking the number of conversation turns.
        system_start (str): Token sequence for starting system messages.
        user_start (str): Token sequence for starting user messages.
        assistant_start (str): Token sequence for starting assistant messages.
        eot (str): End-of-text token.

    Args:
        model_family (str): The model family identifier.
        system_prompt (Optional[str]): Custom system prompt. If None, uses default from SYS_PROMPTS.

    Example:
        >>> builder = Llama3ChatPromptBuilder("llama3", "You are a helpful assistant.")
        >>> builder.add_turn("human", "Hello!")
        '<|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>'
    """
    
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt3(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # Llama-3 Specific tokens
        self.system_start = "<|start_header_id|>system<|end_header_id|>\n"
        self.user_start = "<|start_header_id|>user<|end_header_id|>\n"
        self.assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n"
        self.eot = "<|eot_id|>"

        # Get role-specific "wrap" functions
        self.wrap_system = lambda msg: f"{self.system_start}{msg}{self.eot}"
        self.wrap_human = lambda msg: f"{self.user_start}{msg}{self.eot}"
        self.wrap_gpt = lambda msg: f"{self.assistant_start}{msg}{self.eot}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0


    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for first turn with system prompt
        if self.turn_count == 0:
            # Add system prompt if it exists
            wrapped_message = ""
            if self.system_prompt:
                wrapped_message = self.wrap_system(self.system_prompt)
            # Add the human message
            wrapped_message += self.wrap_human(message)
        elif (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for first turn with system prompt
        if self.turn_count == 0:
            # Add system prompt if it exists
            if self.system_prompt:
                prompt_copy += self.wrap_system(self.system_prompt)
            # Add the human message
            prompt_copy += self.wrap_human(message)
        else:
            prompt_copy += self.wrap_human(message)

        return prompt_copy

    def get_prompt(self) -> str:
        return self.prompt


# class Llama3ChatPromptBuilder(PromptBuilder):
#     def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
#         super().__init__(model_family, system_prompt)
#         self.system_prompt = format_system_prompt(
#             SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
#         )

#         # Llama-2 Specific
#         self.bos, self.eos = "<s>", "</s>"

#         # Get role-specific "wrap" functions
#         self.wrap_human = lambda msg: f"[INST] {msg} [/INST] "
#         self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

#         # === `self.prompt` gets built up over multiple turns ===
#         self.prompt, self.turn_count = "", 0

#     def add_turn(self, role: str, message: str) -> str:
#         assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
#         message = message.replace("<image>", "").strip()

#         # Special Handling for "system" prompt (turn_count == 0)
#         if self.turn_count == 0:
#             sys_message = self.wrap_human(self.system_prompt + message)
#             wrapped_message = sys_message
#         elif (self.turn_count % 2) == 0:
#             human_message = self.wrap_human(message)
#             wrapped_message = human_message
#         else:
#             gpt_message = self.wrap_gpt(message)
#             wrapped_message = gpt_message

#         # Update Prompt
#         self.prompt += wrapped_message

#         # Bump Turn Counter
#         self.turn_count += 1

#         # Return "wrapped_message" (effective string added to context)
#         return wrapped_message

#     def get_potential_prompt(self, message: str) -> None:
#         # Assumes that it's always the user's (human's) turn!
#         prompt_copy = str(self.prompt)

#         # Special Handling for "system" prompt (turn_count == 0)
#         if self.turn_count == 0:
#             sys_message = self.wrap_human(self.system_prompt + message)
#             prompt_copy += sys_message

#         else:
#             human_message = self.wrap_human(message)
#             prompt_copy += human_message

#         return prompt_copy.removeprefix(self.bos).rstrip()

#     def get_prompt(self) -> str:
#         # Remove prefix <bos> because it gets auto-inserted by tokenizer!
#         return self.prompt.removeprefix(self.bos).rstrip()

from typing import List
from abc import ABC, abstractmethod

class BaseModelCaller(ABC):
    """
    Base class for multimodal model calling.

    Users must subclass this and implement the `call()` method
    to define how to send prompt and images to their model or API.
    """
    @abstractmethod
    def call(self, images: List[str], prompt: str) -> str:
        """
        Send prompt and images to model and return its response.

        Args:
            images (List[str]): List of base64-encoded image strings (without MIME header).
            prompt (str): Instructional text to guide the model's behavior.

        Returns:
            str: Model-generated response.
        """
        pass

def call_model(
    images: List[str],
    prompt: str,
    caller: BaseModelCaller
) -> str:
    """
    Args:
        image_base64_list (List[str]):
            A list of image strings encoded in base64 (without the MIME prefix).
            Each item should be a single image in base64 format.
        prompt_content (str):
            A natural language instruction or question that guides the model's response,
            such as "Describe the image" or "What is this UI?".

    Returns:
        str:
            The textual output returned by the model, typically a natural language answer.
    """
    return caller.call(images, prompt)

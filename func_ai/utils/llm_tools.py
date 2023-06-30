import json
import logging
import os
import traceback
from abc import abstractmethod
from enum import Enum
from typing import Any

import openai
from tenacity import retry, wait_fixed, stop_after_attempt

from pydantic import BaseModel, Field

from func_ai.utils.py_function_parser import type_mapping

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ConversationStore(BaseModel):
    """
    A class for storing conversations

    """
    conversation: list = Field(default_factory=list, description="A dictionary of conversations")

    @abstractmethod
    def add_message(self, message: Any):
        """
        Adds a message to the conversation
        :param message:
        :return:
        """
        raise NotImplementedError

    def get_conversation(self) -> list:
        """
        Returns the conversation
        :return:
        """
        return self.conversation


class OpenAIConversationStore(ConversationStore):

    def add_message(self, message: Any):
        self.conversation.append(message)

    def add_system_message(self, message: str):
        # remove any existing system messages
        self.conversation = [x for x in self.conversation if x['role'] != 'system']
        self.conversation.insert(0, {"role": "system", "content": message})


class LLMInterface(BaseModel):
    """
    Interface for interacting with the Language Learning Model

    """
    usage: dict = Field(default_factory=dict, description="A dictionary of the usage of the API")
    cost_mapping: dict = Field(default_factory=dict, description="A dictionary of the cost of the API")
    conversation_store: ConversationStore = Field(...,
                                                  description="A class for storing conversations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def send(self, prompt: str, **kwargs) -> Any:
        """
        Sends a prompt to the API

        :param prompt:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def load_cost_mapping(self, file_path: str) -> None:
        """
        Loads the cost mapping from a file

        :param file_path:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def update_cost(self, model, api_response) -> None:
        """
        Updates the cost of the API

        :param model: The model used
        :param api_response: The response from the API
        :return:
        """
        raise NotImplementedError

    def get_usage(self) -> dict:
        return self.usage

    def reset_usage(self) -> None:
        """
        Resets the usage of the API

        :return:
        """
        self.usage = dict()

    def store_usage(self, file_path: str) -> None:
        """
        Appends the usage of the API to a file

        :param file_path: The path to the file
        :return:
        """
        with open(file_path, "a") as f:
            json.dump(self.usage, f)

    def get_conversation(self) -> list[any]:
        """
        Returns the conversation
        :return:
        """
        return self.conversation_store.get_conversation()


class OpenAIInterface(LLMInterface):
    """
    Interface for interacting with the OpenAI API
    """
    max_tokens: int = Field(default=256, description="The maximum number of tokens to return")
    model: str = Field(default="gpt-3.5-turbo-0613", description="The model to use")
    temperature: float = Field(default=0.0, description="The temperature to use")
    conversation_store: OpenAIConversationStore = Field(default_factory=OpenAIConversationStore,
                                                        description="A class for storing conversations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_tokens=kwargs.get("max_tokens", 256),
                         model=kwargs.get("model", "gpt-3.5-turbo-0613"),
                         temperature=kwargs.get("temperature", 0.0),
                         conversation_store=OpenAIConversationStore())
        openai.api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(1),
           retry_error_callback=lambda x: logger.warning(x))
    def send(self, prompt: str, **kwargs) -> dict:
        _functions = kwargs.get("functions", None)
        _model = kwargs.get("model", self.model)
        # print(type(self._conversation_store))
        self.conversation_store.add_message({"role": "user", "content": prompt})
        logger.debug(f"Prompt: {prompt}")
        try:
            if _functions:
                response = openai.ChatCompletion.create(
                    model=_model,
                    messages=self.conversation_store.get_conversation(),
                    functions=_functions,
                    function_call="auto",
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            else:
                response = openai.ChatCompletion.create(
                    model=_model,
                    messages=self.conversation_store.get_conversation(),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                # decode utf-8 bytes to unicode
            if "content" in response["choices"][0]["message"] and response["choices"][0]["message"]["content"]:
                response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].encode(
                    'utf-8').decode(
                    "utf-8")
            if "function_call" in response["choices"][0]["message"] and response["choices"][0]["message"][
                "function_call"]:
                response["choices"][0]["message"]["function_call"]["arguments"] = \
                    response["choices"][0]["message"]["function_call"]["arguments"].encode('utf-8').decode("utf-8")
            self.update_cost(_model, response)
            _response_message = response["choices"][0]["message"]
            self.conversation_store.add_message(_response_message)
            return _response_message
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            raise e

    def load_cost_mapping(self, file_path: str) -> None:
        with open(file_path) as f:
            self.cost_mapping = json.load(f)

    def update_cost(self, model, api_response) -> None:
        if model not in self.usage:
            self.usage[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        self.usage[model]["prompt_tokens"] += api_response['usage']['prompt_tokens']
        self.usage[model]["completion_tokens"] += api_response['usage']['completion_tokens']
        self.usage[model]["total_tokens"] += api_response['usage']['total_tokens']



class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.schema()

        return {
            "name": schema["title"],
            "description": schema.get("description", f"{schema['title']} class"),
            "parameters": {
                "type": "object",
                "properties": {f"{name}": OpenAISchema.get_field_def(name, field_info) for name, field_info in
                               cls.__fields__.items()},
            },
            "required": [name for name, field_info in cls.__fields__.items() if field_info.required]
        }

    @staticmethod
    def get_field_def(name, field_info) -> dict[str, str]:
        """
        Returns a string representation of a field definition

        :param name:
        :param field_info:
        :return:
        """

        default = f". Default value: {str(field_info.default)}" if not field_info.required else ""
        description = field_info.field_info.description
        if description:
            description = description.replace("\n", " ")
        else:
            description = ""
        _enum_values = ""
        if issubclass(field_info.outer_type_, Enum):
            _enum_values = ". Enum: " + ",".join([f"{_enum.name}" for _enum in field_info.outer_type_])
        return {
            "description": f"{description}{default}{_enum_values}",
            "type": type_mapping(field_info.outer_type_)
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        """
        Returns an instance of the class from LLM completion response

        :param completion: completion response from LLM
        :param throw_error:  whether to throw error if function call is not present
        :return:
        """
        if throw_error:
            assert "function_call" in completion, "No function call detected"
            assert (
                    completion["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = completion["function_call"]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)

    @classmethod
    def from_prompt(cls, prompt: str, llm_interface: LLMInterface, throw_error=True):
        """
        Returns an instance of the class from LLM prompt

        :param prompt: User prompt
        :param llm_interface: LLM interface
        :param throw_error: whether to throw error if function call is not present
        :return:
        """
        completion = llm_interface.send(prompt, functions=[cls.openai_schema])
        # print(llm_interface.get_conversation())
        # TODO add crud interface functions here
        return cls.from_response(completion, throw_error)

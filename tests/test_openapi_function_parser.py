import json

from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import OpenAPISpecOpenAIWrapper


def test_func_wrapper():
    """
    This function tests the func_wrapper
    :return:
    """
    load_dotenv()
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())
    print(_spec.from_prompt("Get pet with id 10", "getPetById").last_call)


def test_func_wrapper_chaining():
    """
    This function tests the func_wrapper
    :return:
    """
    load_dotenv()
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())
    _calls = _spec.from_prompt(
        "Add new pet named Rocky with following photoUrl: http://rocky.me/pic.png. Tag the Rocky with 'dog' and 'pet'",
        "addPet").from_prompt("Get pet", "getPetById").calls
    print(json.dumps(_calls))


def test_get_spec_operation():
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())
    _spec_dict = _spec.to_dict()
    assert "getPetById" in _spec_dict
    assert "addPet" in _spec_dict
    print(json.dumps(_spec.to_dict(), indent=2))

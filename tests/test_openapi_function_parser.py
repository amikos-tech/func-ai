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


def test_spec_qa_basic():
    load_dotenv()
    _spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                              llm_interface=OpenAIInterface())
    _spec_dict = _spec.to_dict()
    #
    # llm_interface = OpenAIInterface()
    # llm_interface.conversation_store.add_system_message("You are an API expert. Your goal is to assist the user"
    #                                                     "in using an API that he/she is not familiar with."
    #                                                     "The user will provide commands which you will use to find out information about an API"
    #                                                     "Then the user will ask you questions about the API and you will answer them."
    #                                                     ""
    #                                                     "Rules:"
    #                                                     "1. You will only answer questions about the API"
    #                                                     "2. You will keep your output only to the essential information")
    # _resp = llm_interface.send(f"Give me an example of how to add a pet\n Context: {_spec.get_operation('addPet')}")
    # print(_resp)
    print("\n".join([f"- {k}: {v}" for k, v in _spec.operations_summary.items()]))

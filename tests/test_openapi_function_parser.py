import json

from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import get_spec_from_url, parse_spec, call_api


def test_call_api():
    load_dotenv()
    spec = get_spec_from_url('http://petstore.swagger.io/v2/swagger.json')
    _funcs = parse_spec(spec)
    print(f"Function to all {_funcs['getPetById']}")
    inf = OpenAIInterface()
    resp = inf.send(
        # prompt="Add new pet named Rocky with following photoUrl: http://rocky.me/pic.png. Tag the Rocky with 'dog' and 'pet'",
        prompt="Get pet with id 10",
        functions=[_funcs['getPetById']['func']])
    print(f"LLM Response: {resp}")
    if "function_call" in resp:
        fc = resp["function_call"]
        api_resp = call_api(fc, spec, _funcs)
        assert api_resp["status_code"] == 200
        assert 'id' in json.loads(api_resp["response"])
        assert json.loads(api_resp["response"])['id'] == 10

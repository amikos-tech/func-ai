# AI Functional Catalog

Your OpenAI function calling on steroids

Features:

- Index any python function and use it in your AI workflows
- Index any CLI command and use it in your AI workflows
- Index any API endpoint and use it in your AI workflows

## Installation

With pip:

```bash
pip install func-ai
```

With poetry:

```bash
poetry add func-ai
```

## Usage

### Pydantic Class Mapping

```python
from pydantic import Field

from func_ai.utils.llm_tools import OpenAIInterface, OpenAISchema


class User(OpenAISchema):
    """
    This is a user
    """
    id: int = Field(None, description="The user's id")
    name: str = Field(..., description="The user's name")


def test_user_openai_schema():
    print(User.from_prompt(prompt="Create a user with id 100 and name Jimmy", llm_interface=OpenAIInterface()).json())
    """
    Returns: {"id": 100, "name": "Jimmy"}
    """

```

### OpenAPI Mapping

```python
import json

from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import get_spec_from_url, parse_spec, call_api

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
    call_api(fc, spec, _funcs)
"""
Function to all {'details': {'name': 'getPetById', 'description': 'Returns a single pet', 'parameters': [{'name': 'petId', 'in': 'path', 'description': 'ID of pet to return', 'required': True, 'type': 'integer', 'format': 'int64'}], 'summary': 'Find pet by ID', 'method': 'get', 'path': '/pet/{petId}'}, 'func': {'name': 'getPetById', 'description': 'Find pet by IDReturns a single pet', 'parameters': {'type': 'object', 'properties': {'petId': {'description': 'ID of pet to return', 'type': 'string', 'in': 'path'}}, 'required': ['petId']}}}
LLM Response: {
"role": "assistant",
"content": null,
"function_call": {
 "name": "getPetById",
 "arguments": "{\n  \"petId\": \"10\"\n}"
}
}
Calling getPetById - get https://petstore.swagger.io/v2/pet/{petId}
2023-06-30 07:57:40,306 - urllib3.connectionpool - DEBUG - https://petstore.swagger.io:443 "GET /v2/pet/10 HTTP/1.1" 200 None
{"id":10,"category":{"id":10,"name":"sample string"},"name":"doggie","photoUrls":[],"tags":[],"status":"pending"}
"""
```

> Note: the above is pretty naive implementation of OpenAPI parsing and calling. It is not production ready.

### Jinja2 Templating

```python
from dotenv import load_dotenv
from func_ai.utils.jinja_template_functions import JinjaOpenAITemplateFunction
from func_ai.utils.llm_tools import OpenAIInterface
load_dotenv()
ji = JinjaOpenAITemplateFunction.from_string_template("Name: {{ NAME }} \n Age: {{ AGE }}", OpenAIInterface())
resp = ji.render_from_prompt("John is 20 years old")
assert "Name: John" in resp
assert "Age: 20" in resp
# prints
"""
Name: John 
Age: 20
"""
```

## Inspiration

- https://github.com/jxnl/openai_function_call
- https://github.com/rizerphe/openai-functions
- https://github.com/aurelio-labs/funkagent
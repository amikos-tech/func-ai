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
from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.openapi_function_parser import OpenAPISpecOpenAIWrapper

load_dotenv()
_spec = OpenAPISpecOpenAIWrapper.from_url('http://petstore.swagger.io/v2/swagger.json',
                                          llm_interface=OpenAIInterface())
print(_spec.from_prompt("Get pet with id 10", "getPetById").last_call)
"""
2023-07-03 10:43:04 DEBUG Starting new HTTP connection (1): petstore.swagger.io:80
2023-07-03 10:43:04 DEBUG http://petstore.swagger.io:80 "GET /v2/swagger.json HTTP/1.1" 301 134
2023-07-03 10:43:04 DEBUG Starting new HTTPS connection (1): petstore.swagger.io:443
2023-07-03 10:43:04 DEBUG https://petstore.swagger.io:443 "GET /v2/swagger.json HTTP/1.1" 200 None
2023-07-03 10:43:04 DEBUG Prompt: Get pet with id 10
2023-07-03 10:43:04 DEBUG message='Request to OpenAI API' method=post path=https://api.openai.com/v1/chat/completions
2023-07-03 10:43:04 DEBUG api_version=None data='{"model": "gpt-3.5-turbo-0613", "messages": [{"role": "user", "content": "Get pet with id 10"}], "functions": [{"name": "getPetById", "description": "Find pet by IDReturns a single pet", "parameters": {"type": "object", "properties": {"petId": {"description": "ID of pet to return", "type": "string", "in": "path"}}, "required": ["petId"]}}], "function_call": "auto", "temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_tokens": 256}' message='Post details'
2023-07-03 10:43:04 DEBUG Converted retries value: 2 -> Retry(total=2, connect=None, read=None, redirect=None, status=None)
2023-07-03 10:43:05 DEBUG Starting new HTTPS connection (1): api.openai.com:443
2023-07-03 10:43:06 DEBUG https://api.openai.com:443 "POST /v1/chat/completions HTTP/1.1" 200 None
2023-07-03 10:43:06 DEBUG message='OpenAI API response' path=https://api.openai.com/v1/chat/completions processing_ms=876 request_id=f38d1625ae785681b53686492fd1d7e3 response_code=200
2023-07-03 10:43:06 DEBUG Starting new HTTPS connection (1): petstore.swagger.io:443
2023-07-03 10:43:07 DEBUG https://petstore.swagger.io:443 "GET /v2/pet/10 HTTP/1.1" 200 None
PASSED                                                                   [100%]{'function_call': <OpenAIObject at 0x10a5f2c30> JSON: {
  "name": "getPetById",
  "arguments": "{\n  \"petId\": \"10\"\n}"
}, 'function_response': {'role': 'function', 'name': 'getPetById', 'content': '{\'status_code\': 200, \'response\': \'{"id":10,"category":{"id":10,"name":"sample string"},"name":"doggie","photoUrls":["sample 1","sample 2","sample 3"],"tags":[{"id":10,"name":"sample string"},{"id":10,"name":"sample string"}],"status":"available"}\'}'}}

"""
```

> Note: The above example is still in beta and is not production ready.

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
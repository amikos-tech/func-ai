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

### Class Mapping

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
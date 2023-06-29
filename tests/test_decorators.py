from dotenv import load_dotenv
from pydantic import Field

from func_ai.utils.llm_tools import OpenAIInterface, OpenAISchema

load_dotenv()


class User(OpenAISchema):
    """
    This is a user
    """
    id: int = Field(None, description="The user's id")
    name: str = Field(..., description="The user's name")


def test_user_openai_schema():
    print(User.from_prompt(prompt="Create a user with id 100 and name Jimmy", llm_interface=OpenAIInterface()).json())

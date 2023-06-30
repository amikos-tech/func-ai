from enum import Enum

from dotenv import load_dotenv
from pydantic import Field

from func_ai.utils.llm_tools import OpenAIInterface, OpenAISchema

load_dotenv()


class UserType(str, Enum):
    """
    This is a user type
    """
    patient = "patient"
    doctor = "doctor"
    nurse = "nurse"


class User(OpenAISchema):
    """
    This is a user
    """
    id: int = Field(None, description="The user's id")
    name: str = Field(..., description="The user's name")
    type: UserType = Field(default="doctor", description="The user's type")


def test_from_prompt():
    # print(User.openai_schema)
    _user = User.from_prompt(prompt="Create a user with id 100 and name Jimmy. Jimmy is a nurse",
                             llm_interface=OpenAIInterface()).dict()
    print(_user)
    assert _user["id"] == 100
    assert _user["name"] == "Jimmy"
    assert _user["type"] == "nurse"

# Python Function Calling

In this article we'll cover how to call Python functions using `func-ai`.

## Pre-requisites

Before you begin make sure you have the following:

- `func-ai` installed (`pip install func-ai`)
- An OpenAI API key set in the `OPENAI_API_KEY` environment variable (you can have a `.env` file in the current
  directory with `OPENAI_API_KEY=<your-api-key>` and then use load_dotenv() to load the environment variables from the
  file)

## Calling Python functions using OpenAI API

First let's define a python function we want to call using LLM:

```python
def add_two_numbers(a: int, b: int) -> int:
    """
    Adds two numbers
    
    :param a: The first number
    :param b: The second number
    :return: The sum of the two numbers
    """
    return a + b
```

A few key points about how functions we want to expose to LLMs should be defined:

- The function MUST have type-hints for all parameters and the return value. This helps LLMs understand what the
  function does and how to call it.
- The function MUST have a docstring. The docstring and in particular the description is used by the LLM to identify the
  function to call.
- The function docstring MUST contain parameters and their descriptions. This helps LLMs understand what parameters the
  function takes and what they are used for.

Now let's convert the above function so that it can be called using OpenAI function calling capability:

```python
from func_ai.utils.py_function_parser import func_to_json

_json_fun = func_to_json(add_two_numbers)
```

In the above snippet we use `func_to_json` to convert the python function to a dictionary that can be passed to OpenAI
API.

Now let's do some prompting to see how the function can be called:

```python
import openai
import json
from dotenv import load_dotenv

load_dotenv()


def call_openai(_messages, _functions: list = None):
    if _functions:
        _open_ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=_messages,
            functions=_functions,
            function_call="auto",
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=256,
        )
    else:
        _open_ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=_messages,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=256,
        )
    return _open_ai_resp["choices"][0]["message"]


_messages = [{"role": "system",
              "content": "You are a helpful automation system that helps users to perform a variety of supported tasks."},
             {"role": "user", "content": "I want to add 5 and 10"}]
_functions = [_json_fun]
response = call_openai(_messages, _functions)
if "function_call" in response:
    _result = add_two_numbers(**json.loads(response["function_call"]["arguments"]))
    print(f"Result: {_result}")
    _function_call_llm_response = {
        "role": "function",
        "name": response["function_call"]["name"],
        "content": f"Result: {_result}",
    }
    _messages.append(_function_call_llm_response)
    print(call_openai(_messages))
```

The above snippet will print the following:

```text
Result: 15
{
  "role": "assistant",
  "content": "The sum of 5 and 10 is 15."
}
```

Let's break down the above snippet:

- First we define a function `call_openai` that takes a list of messages and a list of functions to call. The function
  uses the `openai.ChatCompletion.create` API to call OpenAI and get a response.
- Next we define a list of messages that we want to send to OpenAI. The first message is a system message that describes
  what the system does. The second message is a user message that tells the system what the user wants to do.
- Next we define a list of functions that we want to expose to OpenAI. In this case we only have one function.
- Next we call the `call_openai` function with the messages and functions. The response from OpenAI is stored in the
  `response` variable.
- Next we check if the response contains a `function_call` key. If it does then we know that OpenAI has called our
  function and we can get the result from the `function_call` key.
- Next we print the result of the function call.
- Next we create a new message that contains the result of the function call and append it to the list of messages.
- Finally we call the `call_openai` function again with the updated list of messages. This time OpenAI will respond with
  a message that contains the result of the function call.

!!! note "Non-Production Example"

    The above is a naive example of how you can use the `func-ai` library to convert your python functions and use them
    with OpenAI. `func-ai` offer much more advanced mechanisms to help you build a production ready code. Please check
    other articles in the documentation to learn more or get in touch [with us](mailto:info@amikos.tech) if you need help.

## Working with `functools.partial`

Python `functools` library offers the ability to create partial functions with some of the parameters already set. This
is particularly useful in cases where you have either static parameter you want to configure, sensitive parameter such a
secret or a state object (e.g. DB connection) in which case you either cannot or do not want to send that info to
OpenAI. `partial` to the rescue!

Let's create a new function called `query_db` where we want our DB driver to be a fixed parameter and not passed to the
LLM:

> Note: We make the assumption that `call_openai` function is already defined as per the previous example.

```python
from functools import partial
from func_ai.utils.py_function_parser import func_to_json
import json


def query_db(db_driver: object, query: str) -> str:
    """
    Queries the database
    
    :param db_driver: The database driver to use
    :param query: The query to execute
    :return: The result of the query
    """
    return f"Querying {db_driver} with query {query}"


_partial_fun = partial(query_db, db_driver="MySQL")
_json_fun = func_to_json(_partial_fun)
_messages = [{"role": "system",
              "content": "You are a helpful automation system that helps users to perform a variety of supported tasks."},
             {"role": "user", "content": "Query the db for quarterly sales."}]
_functions = [_json_fun]
response = call_openai(_messages, _functions)
if "function_call" in response:
    _result = _partial_fun(**json.loads(response["function_call"]["arguments"]))
    print(f"Result: {_result}")
    _function_call_llm_response = {
        "role": "function",
        "name": response["function_call"]["name"],
        "content": f"Result: {_result}",
    }
    _messages.append(_function_call_llm_response)
    print(call_openai(_messages))
```

The above snippet will print the following:

```text
Result: Querying MySQL with query SELECT * FROM sales WHERE date >= '2021-01-01' AND date <= '2021-12-31'
{
  "role": "assistant",
  "content": "Here are the quarterly sales for the year 2021:\n\n1st Quarter: $XXX\n2nd Quarter: $XXX\n3rd Quarter: $XXX\n4th Quarter: $XXX\n\nPlease let me know if there's anything else I can assist you with!"
}
```

The example above is very similar to our previous example except that this time we have fixed the `db_driver` parameter
which gives you that very important security and privacy aspect especially when playing around with LLMs on the open
internet.
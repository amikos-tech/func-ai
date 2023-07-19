# Function Indexing

The library supports function indexing (with some limitations). This means that you can index your functions and then
query them using the `func-ai` library. This is useful if you want to query your functions using natural language and
especially when you have a lot of functions which cannot fit in LLM context.

The Function Indexer (FI) relies on chromadb vector store to store function descriptions and then perform semantic
search on those descriptions to find the most relevant functions.

Limitations:

- partials while supported for indexing and function wrapping using `OpenAIFunctionWrapper` cannot be rehydrated in the
  index once it is reloaded (e.g. app restart). The suggested workaround is at app/script startup to reindex the
  partials which will not re-add them in the index but will only rehydrate them in the index map.

## Usage

```python

import chromadb
from chromadb import Settings
from dotenv import load_dotenv

from func_ai.function_indexer import FunctionIndexer


def function_to_index(a: int, b: int) -> int:
    """
    This is a function that adds two numbers
  
    :param a: First number
    :param b: Second number
    :return: Sum of a and b
    """
    return a + b


def another_function_to_index() -> str:
    """
    This is a function returns hello world
  
    :return: Hello World
    """

    return "Hello World"


def test_function_indexer_init_no_args_find_function_enhanced_summary():
    load_dotenv()
    _indexer = FunctionIndexer(chroma_client=chromadb.PersistentClient(settings=Settings(allow_reset=True)))
    _indexer.reset_function_index()
    _indexer.index_functions([function_to_index, another_function_to_index], enhanced_summary=True)
    _results = _indexer.find_functions("Add two numbers", max_results=10, similarity_threshold=0.2)
    assert len(_results) == 1
    assert _results[0].function(1, 2) == 3


if __name__ == "__main__":
    test_function_indexer_init_no_args_find_function_enhanced_summary()
```

The above code shows how to use the two main functions of the Function Indexer:

- `index_functions` which indexes a list of functions
- `find_functions` which finds functions based on a query string

## API Docs

### FunctionIndexer

Init args:

- `chroma_client`: A chromadb client to use for storing the function index. If not provided a new client will be created
  using the default settings (e.g. `chromadb.PersistentClient(settings=Settings(allow_reset=True))`).
- `llm_interface`: An LLM interface to use for function wrapping. If not provided a new LLM interface will be created
  using the default settings (e.g. `OpenAIInterface()`).
- `embedding_function`: A function that takes a string and returns an embedding. If not provided the default embedding
  function will be used (e.g. `embedding_functions.OpenAIEmbeddingFunction()`).
- `collection_name`: The name of the collection to use for storing the function index. If not provided the defaults
  to `function_index`.

> Note: You should always initialize your FunctionIndexer with the same embedding function

#### `index_functions`

Args:

- `functions`: A list of functions to index
- `enhanced_summary`: If True the function summary will be enhanced with the function docstring. Defaults to False.
- `llm_interface`: An LLM interface to use for function wrapping. If not provided the one used in Indexer init will be
  used

#### `find_functions`

Args:

- `query`: The query string to use for finding functions
- `max_results`: The maximum number of results to return. Defaults to 2.
- `similarity_threshold`: The similarity threshold to use for filtering results. Defaults to 1.0.

Returns a named tuple `SearchResult` with the following fields:

- `function`: The function actual function that can be directly called
- `name`: The function name
- `wrapper`: The `OpenAIFunctionWrapper` function wrapper
- `distance`: The distance of the function from the query string

> Note: The returned list is sorted by distance in ascending order (e.i. the first result is the closest to the query)

#### `functions_summary`

Returns: A dictionary containing function names and their descriptions.

#### `index_wrapper_functions`

This identical to `index_functions` but the list of functions is a list of `OpenAIFunctionWrapper` objects.
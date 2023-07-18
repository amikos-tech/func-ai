import inspect
import logging
import os

import chromadb
import openai
from chromadb import Settings
from chromadb.utils import embedding_functions
from ulid import ULID

from func_ai.utils.llm_tools import OpenAIFunctionWrapper
from func_ai.utils.py_function_parser import func_to_json

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FunctionIndexer(object):
    """
    Index functions
    """

    def __init__(self, db_path: str, collection_name: str = "function_index", **kwargs) -> None:
        """
        Initialize function indexer
        :param db_path: The path where to store the database
        :param collection_name: The name of the collection
        :param kwargs: Additional arguments
        """
        self._client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path  # Optional, defaults to .chromadb/ in the current directory
        ))
        openai.api_key = kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        self.collection_name = collection_name
        self._fns_map = {}
        self._fns_index_map = {}
        self._open_ai_function_map = []
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002"
        )

    def reset_function_index(self) -> None:
        """
        Reset function index

        :return:
        """

        self._client.reset()

    def index_functions(self, functions: list[callable]) -> None:
        """
        Index one or more functions
        Note: Function uniqueness is not checked in this version

        :param functions:
        :return:
        """

        _ai_fun_map, _fns_map, _fns_index_map = FunctionIndexer.get_functions(functions)
        collection = self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"},
                                                           embedding_function=self.openai_ef)
        self._fns_map.update(_fns_map)
        self._open_ai_function_map.extend(_ai_fun_map)
        self._fns_index_map.update(_fns_index_map)
        collection.add(documents=[f['description'] for f in _fns_index_map.values()],
                       metadatas=[{"name": f,
                                   "file": str(inspect.getfile(_fns_map[f])),
                                   "module": inspect.getmodule(_fns_map[f]).__name__} for f, v in
                                  _fns_index_map.items()],
                       ids=[str(ULID()) for _ in _fns_index_map.values()])

    def index_wrapper_functions(self, functions: list[OpenAIFunctionWrapper]):
        """
        Index one or more functions
        Note: Function uniqueness is not checked in this version
        :param functions:
        :return:
        """
        collection = self._client.get_or_create_collection(name=self.collection_name,
                                                           metadata={"hnsw:space": "cosine"},
                                                           embedding_function=self.openai_ef)
        # print(f"Docs: {collection.get()}")
        collection.add(documents=[f.description for f in functions],
                       metadatas=[{"name": f.name, **f.metadata_dict} for f in
                                  functions],
                       ids=[str(ULID()) for _ in functions])

    def rehydrate_function_map(self, functions: list[callable]) -> None:
        """
        Rehydrate function map

        :param functions:
        :return:
        """

        _ai_fun_map, _fns_map, _fns_index_map = FunctionIndexer.get_functions(functions)
        self._fns_map.update(_fns_map)
        self._open_ai_function_map.extend(_ai_fun_map)
        self._fns_index_map.update(_fns_index_map)

    def get_ai_fn_abbr_map(self) -> dict[str, str]:
        """
        Get AI function abbreviated map

        :return: Map of function name (key) and description (value)
        """

        return {f['name']: f['description'] for f in self._open_ai_function_map}

    def find_functions(self, query: str, max_results: int = 2, similarity_threshold: float = 1.0) -> callable:
        """
        Find functions by description

        :param query: Query string
        :param max_results: Maximum number of results
        :param similarity_threshold: Similarity threshold - a cut-off threshold for the similarity score - default is 1.0 (very loose match)
        :return:
        """
        _response = []
        collection = self._client.get_or_create_collection(name=self.collection_name,
                                                           metadata={"hnsw:space": "cosine"},
                                                           embedding_function=self.openai_ef)
        # print(collection.get())
        res = collection.query(query_texts=[query], n_results=max_results)
        print(f"Got results from sematic search: {res}")
        for r in range(len(res['documents'][0])):
            print(f"Distance: {res['distances'][0][r]} vs threshold: {similarity_threshold}")
            if res['distances'][0][r] <= similarity_threshold:
                _response.append(res['metadatas'][0][r]['name'])
        return _response

    @staticmethod
    def get_functions(functions: list[callable]) -> (list, dict):
        """
        Get functions and function map.

        Note: Right now this is a naive implementation as it ignores modules and file paths.

        :param functions:  List of functions
        :return: List of converted functions and function map
        """

        _converted_functions = [func_to_json(_f) for _f in functions]
        _function_map = {f.__name__: f for f in functions}
        _function_index_map = {f.__name__: func_to_json(f) for f in functions}
        return _converted_functions, _function_map, _function_index_map

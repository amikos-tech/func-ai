"""
Function indexer module is responsible for making functions searchable
"""
import importlib
import inspect
import logging
import os
from collections import namedtuple

import chromadb
import openai
from chromadb import Settings
from chromadb.api import EmbeddingFunction
from chromadb.utils import embedding_functions

from func_ai.utils.llm_tools import OpenAIFunctionWrapper, OpenAIInterface, LLMInterface
from func_ai.utils.py_function_parser import func_to_json

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

SearchResult = namedtuple('SearchResult', ['name', 'wrapper', 'function', 'distance'])


class FunctionIndexer(object):
    """
    Index functions
    """

    def __init__(self, llm_interface: LLMInterface = OpenAIInterface(),
                 chroma_client: chromadb.Client = chromadb.PersistentClient(settings=Settings(allow_reset=True)),
                 embedding_function: EmbeddingFunction = None,
                 collection_name: str = "function_index", **kwargs) -> None:
        """
        Initialize function indexer
        :param db_path: The path where to store the database
        :param collection_name: The name of the collection
        :param kwargs: Additional arguments
        """
        # self._client = chromadb.PersistentClient(path=db_path, settings=Settings(
        #     anonymized_telemetry=False,
        #     allow_reset=True,
        # ))
        self._client = chroma_client
        openai.api_key = kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        if embedding_function is None:
            self._embedding_function = embedding_functions.OpenAIEmbeddingFunction()
        else:
            self._embedding_function = embedding_function
        self.collection_name = collection_name
        self._init_collection()

        self._llm_interface = llm_interface
        self._fns_map = {}
        self._fns_index_map = {}
        self._open_ai_function_map = []
        self._functions = {}
        _get_results = self._collection.get()
        if _get_results is not None:
            for idx, m in enumerate(_get_results['metadatas']):
                if "is_partial" in m and bool(m["is_partial"]):
                    logger.warning(
                        f"Found partial function {m['name']}. This function will not be rehydrated into the index.")
                    continue
                self._functions[m["hash"]] = OpenAIFunctionWrapper.from_python_function(
                    func=FunctionIndexer.function_from_ref(m["identifier"]), llm_interface=self._llm_interface)

    def _init_collection(self) -> None:
        self._collection = self._client.get_or_create_collection(name=self.collection_name,
                                                                 metadata={"hnsw:space": "cosine"},
                                                                 embedding_function=self._embedding_function)

    @staticmethod
    def function_from_ref(ref_identifier: str) -> callable:
        """
        Get function from reference
        :param ref_identifier: The reference identifier
        :return: The function
        """
        parts = ref_identifier.split('.')
        _fn = parts[-1]
        _mod = ""
        _last_mod = ""
        _module = None
        for pt in parts[:-1]:
            try:
                _last_mod = str(_mod)
                _mod += pt
                _module = importlib.import_module(_mod)
                _mod += "."
                # module = importlib.import_module('.'.join(parts[:-1]))
                # function = getattr(module, _fn)
            except ModuleNotFoundError:
                print("Last module: ", _last_mod)
                _module = importlib.import_module(_last_mod[:-1] if _last_mod.endswith(".") else _last_mod)
                _module = getattr(_module, pt)
                # print(f"Module: {getattr(module, pt)}")
        # module = importlib.import_module('.'.join(parts[:-1]))
        if _module is None:
            raise ModuleNotFoundError(f"Could not find module {_mod}")
        _fn = _module
        part = parts[-1]
        _fn = getattr(_fn, part)
        return _fn

    def reset_function_index(self) -> None:
        """
        Reset function index

        :return:
        """

        self._client.reset()
        self._init_collection()

    def index_functions(self, functions: list[callable or OpenAIFunctionWrapper],
                        llm_interface: LLMInterface = None,
                        enhanced_summary: bool = False) -> None:
        """
        Index one or more functions
        Note: Function uniqueness is not checked in this version

        :param llm_interface: The LLM interface
        :param functions: The functions to index
        :param enhanced_summary: Whether to use enhanced summary
        :return:
        """
        _fn_llm_interface = llm_interface if llm_interface is not None else self._llm_interface
        _wrapped_functions = [
            OpenAIFunctionWrapper.from_python_function(func=f, llm_interface=_fn_llm_interface) for f
            in functions if not isinstance(f, OpenAIFunctionWrapper)]
        _wrapped_functions.extend([f for f in functions if isinstance(f, OpenAIFunctionWrapper)])
        _fn_hashes = [f.hash for f in _wrapped_functions]
        _existing_fn_results = self._collection.get(ids=_fn_hashes)
        print(_existing_fn_results)
        # filter wrapped functions that are already in the index
        _original_wrapped_functions = _wrapped_functions.copy()
        _wrapped_functions = [f for f in _wrapped_functions if f.hash not in _existing_fn_results["ids"]]
        if len(_wrapped_functions) == 0:
            logger.info("No new functions to index")
            self._functions.update(
                {f.hash: f for f in _original_wrapped_functions})  # we only rehydrate that are already in the index
            return
        _docs = []
        _metadatas = []
        _ids = []
        _function_summarizer = OpenAIInterface(max_tokens=200)
        for f in _wrapped_functions:
            if enhanced_summary:
                _function_summarizer.add_conversation_message(
                    {"role": "system",
                     "content": "You are an expert summarizer. Your purpose is to provide a good summary of the function so that the user can add the summary in an embedding database which will them be searched."})
                _fsummary = _function_summarizer.send(f"Summarize the function below.\n\n{inspect.getsource(f.func)}")
                _docs.append(f"{_fsummary['content']}")
                _function_summarizer.conversation_store.clear()
            else:
                _docs.append(f"{f.description}")
            _metadatas.append(
                {"name": f.name, "identifier": f.identifier, "hash": f.hash, "is_partial": str(f.is_partial),
                 **f.metadata_dict})
            _ids.append(f.hash)

        self._collection.add(documents=_docs,
                             metadatas=_metadatas,
                             ids=_ids)
        self._functions.update({f.hash: f for f in _wrapped_functions})

    def index_wrapper_functions(self, functions: list[OpenAIFunctionWrapper],
                                llm_interface: LLMInterface = None,
                                enhanced_summary: bool = False) -> None:
        """
        Index one or more functions
        Note: Function uniqueness is not checked in this version
        :param functions: The functions to index
        :param llm_interface: The LLM interface
        :param enhanced_summary: Whether to use enhanced summary
        :return: None
        """
        self.index_functions(functions=functions, llm_interface=llm_interface, enhanced_summary=enhanced_summary)

    def get_ai_fn_abbr_map(self) -> dict[str, str]:
        """
        Get AI function abbreviated map

        :return: Map of function name (key) and description (value)
        """

        return {f['name']: f['description'] for f in self._open_ai_function_map}

    def functions_summary(self) -> dict[str, str]:
        """
        Get functions summary

        :return: Map of function name (key) and description (value)
        """
        return {f.name: f.description for f in self._functions.values()}

    def find_functions(self, query: str, max_results: int = 2, similarity_threshold: float = 1.0) -> list[SearchResult]:
        """
        Find functions by description

        :param query: Query string
        :param max_results: Maximum number of results
        :param similarity_threshold: Similarity threshold - a cut-off threshold for the similarity score - default is 1.0 (very loose match)
        :return:
        """
        _response = []
        # print(self._functions.keys())
        res = self._collection.query(query_texts=[query], n_results=max_results)
        # print(f"Got results from sematic search: {res}")
        for idx, _ in enumerate(res['documents'][0]):
            print(f"Distance: {res['distances'][0][idx]} vs threshold: {similarity_threshold}")
            if res['distances'][0][idx] <= similarity_threshold:
                _search_res = SearchResult(name=res['metadatas'][0][idx]['name'],
                                           function=self._functions[res['metadatas'][0][idx]['hash']].func,
                                           wrapper=self._functions[res['metadatas'][0][idx]['hash']],
                                           distance=res['distances'][0][idx])
                _response.append(_search_res)

        _response.sort(key=lambda x: x.distance)
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

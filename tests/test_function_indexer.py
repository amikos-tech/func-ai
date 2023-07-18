import os

import openai
from dotenv import load_dotenv

from func_ai.function_indexer import FunctionIndexer


def fun_to_index_1() -> str:
    """
    This is a function that bars

    :return: Returns something of interest
    """
    pass


def fun_to_index_2() -> str:
    """
    This is a function that foo bar

    :return: Returns nothing of interest
    """
    pass


def test_find_functions_with_threshold_some_results_above():
    """
    This function tests the search with a threshold
    :return:
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _indexer = FunctionIndexer("./fi_test_1")
    _indexer.reset_function_index()
    _indexer.index_functions([fun_to_index_1, fun_to_index_2])
    _res = _indexer.find_functions("Function to foo bar", similarity_threshold=0.1)
    assert len(_res) == 1
    print(f"Got response: {_res}")


def test_find_functions_with_threshold_all_results_below():
    """
    This function tests the search with a threshold
    :return:
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _indexer = FunctionIndexer("./fi_test_1")
    _indexer.reset_function_index()
    _indexer.index_functions([fun_to_index_1, fun_to_index_2])
    _res = _indexer.find_functions("Function to foo bar", similarity_threshold=0.5)
    assert len(_res) == 2
    print(f"Got response: {_res}")

import functools
import inspect
import re


def type_mapping(dtype: type) -> str:
    """
    Map python types to json schema types

    :param dtype:
    :return:
    """
    if dtype == float:
        return "number"
    elif dtype == int:
        return "integer"
    elif dtype == str:
        return "string"
    else:
        return "string"


def extract_params(doc_str: str) -> dict[str, str]:
    """
    Parse the docstring to get the descriptions for each parameter in dict format

    :param doc_str:
    :return:
    """
    # split doc string by newline, skipping empty lines
    params_str = [line for line in doc_str.split("\n") if line.strip()]
    params = {}
    for line in params_str:
        # we only look at lines starting with ':param'
        if line.strip().startswith(':param'):
            param_match = re.findall(r'(?<=:param )\w+', line)
            if param_match:
                param_name = param_match[0]
                desc_match = line.replace(f":param {param_name}:", "").strip()
                # if there is a description, store it
                if desc_match:
                    params[param_name] = desc_match
    return params


def extract_return_description(docstring):
    """
    Extract the return description from a Python docstring.

    :param docstring: The docstring to extract the return description from.
    :return: The return description, or empty string if no return description is found.
    """
    match = re.search(r':return: (.*)', docstring)
    if match:
        return " " + match.group(1)
    else:
        return ""


def func_to_json(func) -> dict[str, any]:
    """
    Convert a function to a json schema

    :param func: Python function
    :return:
    """
    # Check if the function is a functools.partial
    if isinstance(func, functools.partial) or isinstance(func, functools.partialmethod):
        fixed_args = func.keywords
        _func = func.func
        if isinstance(func, functools.partial) and (fixed_args is None or fixed_args == {}):
            fixed_args = dict(zip(func.func.__code__.co_varnames, func.args))
    else:
        fixed_args = {}
        _func = func

    # first we get function name
    func_name = _func.__name__
    # then we get the function annotations
    argspec = inspect.getfullargspec(_func)
    # get the function docstring
    func_doc = inspect.getdoc(_func)
    # parse the docstring to get the description
    func_description = ''.join([line for line in func_doc.split("\n") if not line.strip().startswith(':')])
    func_description += extract_return_description(func_doc)
    # parse the docstring to get the descriptions for each parameter in dict format
    param_details = extract_params(func_doc) if func_doc else {}
    # attach parameter types to params and exclude fixed args
    # get params
    params = {}
    for param_name in argspec.args:
        if param_name not in fixed_args.keys():
            params[param_name] = {
                "description": param_details.get(param_name) or "",
                "type": type_mapping(argspec.annotations.get(param_name, type(None)))
            }
    # calculate required parameters excluding fixed args
    # _required = [arg for arg in argspec.args if arg not in fixed_args]
    _required = [i for i in argspec.args if i not in fixed_args.keys()]
    if inspect.getfullargspec(_func).defaults:
        _required = [argspec.args[i] for i, a in enumerate(argspec.args) if
                     argspec.args[i] not in inspect.getfullargspec(_func).defaults and argspec.args[
                         i] not in fixed_args.keys()]
    # then return everything in dict
    # TODO: Move this to OpenAIFunctionWrapper
    return {
        "name": func_name,
        "description": func_description,
        "parameters": {
            "type": "object",
            "properties": params,
            "required": _required
        },

    }

""" doc """
import requests
import json

from dotenv import load_dotenv

from func_ai.utils.llm_tools import OpenAIInterface


def get_spec_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception in case of a failure
    return response.json()


def get_operations_from_path_item(path_item):
    http_methods = ["get", "put", "post", "delete", "options", "head", "patch", "trace"]
    operations = [{"method": op, "spec": op_spec} for op, op_spec in path_item.items() if op in http_methods]
    return operations


def get_operation_details(operation, path):
    return {
        "name": operation['spec'].get('operationId'),
        "description": operation['spec'].get('description'),
        "parameters": operation['spec'].get('parameters'),
        "summary": operation['spec'].get('summary', ''),
        "method": operation['method'],
        "path": path,
    }


def parse_parameters(parameters, _defs):
    params_dict = {"type": "object", "properties": {}, "required": []}

    for param in parameters:
        name = param['name']
        description = param.get('description', '')  # TODO generate a description if not present
        if 'schema' in param and '$ref' in param['schema']:
            _param_model = param.get('schema').get('$ref').replace('#/definitions/', '')
            param_model = _defs.get(_param_model)
            for p, pd in param_model['properties'].items():
                if '$ref' in pd:
                    param_model['properties'][p] = _defs.get(pd['$ref'].replace('#/definitions/', ''))
                if 'items' in pd and '$ref' in pd['items']:
                    param_model['properties'][p]['items'] = _defs.get(pd['items']['$ref'].replace('#/definitions/', ''))
            description += ".\nJSON Model: " + json.dumps(param_model)
        param_type = param['schema'].get('type', 'string') if 'schema' in param else 'string'
        # TODO if parameter is schema then
        params_dict["properties"][name] = {
            "description": description,
            "type": param_type,
            "in": param['in'] if 'in' in param else "query"
        }

        if param.get('required', False):
            params_dict["required"].append(name)

    return params_dict


def get_func_details(operation, _defs):
    name = operation.get('name')
    description = operation.get('summary', '') + operation.get(
        'description')  # TODO if description not present generate one with AI
    parameters = operation.get('parameters', [])

    # Parse the parameters
    parameters = parse_parameters(parameters, _defs)

    return {
        "name": name,
        "description": description,
        "parameters": parameters
    }


def parse_spec(spec):
    _defs = spec.get('definitions')
    paths = spec['paths']
    _funcs = {}
    for path, path_item in paths.items():
        operations = get_operations_from_path_item(path_item)
        for operation in operations:
            details = get_operation_details(operation, path)
            _funcs[details['name']] = {"details": details, "func": get_func_details(details, _defs)}
    return _funcs


def _coerce_type(value, type):
    if type == "integer":
        return int(value)
    if type == "number":
        return float(value)
    if type == "boolean":
        return bool(value)
    return value


def call_api(llm_function_call: dict, spec: dict, _funcs: dict):
    """
    Calls the API with the given function call

    :param llm_function_call:
    :param spec:
    :param _funcs:
    :return:
    """
    f_name = llm_function_call["name"]
    f_args = json.loads(llm_function_call["arguments"])
    _url = f"https://{spec['host']}{spec['basePath']}{_funcs[f_name]['details']['path']}"
    print(f"Calling {f_name} - {_funcs[f_name]['details']['method']} {_url}")
    _body_params = [param for param in _funcs[f_name]['details']['parameters'] if
                    param['in'] == 'body']
    _path_params = [param for param in _funcs[f_name]['details']['parameters'] if
                    param['in'] == 'path']
    _query_params = [param for param in _funcs[f_name]['details']['parameters'] if
                     param['in'] == 'query']
    if len(_path_params):
        _url = _url.format(
            **{param['name']: _coerce_type(f_args[param['name']], param['type']) for param in _path_params})
    if len(_query_params):
        _url += "?" + "&".join(
            [f"{param['name']}={f_args[param['name']]}" for param in _query_params])
    _body_param_name = None if len(_body_params) == 0 else _body_params[0]['name']
    api_resp = None
    # TODO headers should comply with the spec
    if _funcs[f_name]['details']['method'] == "post":
        api_resp = requests.post(_url, json=json.loads(f_args[_body_param_name]),
                                 headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    if _funcs[f_name]['details']['method'] == "put":
        api_resp = requests.put(_url, json=json.loads(f_args[_body_param_name]),
                                headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    if _funcs[f_name]['details']['method'] == "patch":
        api_resp = requests.patch(_url, json=json.loads(f_args[_body_param_name]),
                                  headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    if _funcs[f_name]['details']['method'] == "get":
        api_resp = requests.get(_url,
                                headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    return {
        "status_code": api_resp.status_code,
        "response": api_resp.text
    }

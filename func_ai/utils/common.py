import inspect


def arg_in_func(func, arg_name):
    # Get the signature of the function
    signature = inspect.signature(func)

    # Get the parameters of the function from the signature
    parameters = signature.parameters

    # Check if the arg_name is in the parameters
    return arg_name in parameters



def to_bool(value):
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() in ("yes", "y", "true", "t", "1"): return True
    if str(value).lower() in ("no", "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"): return False
    raise ValueError('Invalid value for boolean conversion: ' + str(value))
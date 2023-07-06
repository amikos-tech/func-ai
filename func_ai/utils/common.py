import inspect


def arg_in_func(func, arg_name):
    # Get the signature of the function
    signature = inspect.signature(func)

    # Get the parameters of the function from the signature
    parameters = signature.parameters

    # Check if the arg_name is in the parameters
    return arg_name in parameters

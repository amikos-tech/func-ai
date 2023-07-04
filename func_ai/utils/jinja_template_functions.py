"""
Jinja2 Template Functions
"""
import json

from jinja2 import Template, Environment, FileSystemLoader, meta, DictLoader

from func_ai.utils.llm_tools import LLMInterface


class JinjaOpenAITemplateFunction:

    def __init__(self, environment: Environment, llm_interface: LLMInterface, **kwargs):
        self._lm_interface = llm_interface
        self._environment = environment
        if "filters" in kwargs and isinstance(kwargs["filters"], dict):
            self._environment.filters = kwargs["filters"]
        # self._lm_interface.conversation_store.add_system_message("""You are a code helper. Your goal is to help the user to convert a jinja template into a list of parameters.
        #
        # User will provide a jinja template as input.
        #
        # Your task is to extract the jinja parameters and return them in a bullet list.
        #
        # Do not return anything other than the bullet list.
        # Return just the parameter names and nothing else.
        # Return only jinja2 template parameters and nothing else.
        # """)

    def render_from_prompt(self, prompt: str, template_name: str = "template"):
        """
        This function renders the jinja template with the given prompt
        :param prompt:
        :return:
        """
        self._environment.get_template(template_name)
        source = self._environment.loader.get_source(self._environment, template_name)[0]
        ast = self._environment.parse(source)

        # Get the undeclared variables
        _template_vars = meta.find_undeclared_variables(ast)
        _response = self._lm_interface.send(prompt=prompt,
                                            #TODO: Move this to OpenAIFunctionWrapper
                                            functions=[{"name": "render_template",
                                                        "description": "Render a template with given parameters",
                                                        "parameters": {
                                                            "type": "object",
                                                            "properties": {k: {"type": "string",
                                                                               "description": f"{k}"} for
                                                                           k
                                                                           in _template_vars},
                                                            "required": list(
                                                                _template_vars)}}])  # TODO this is a bug should be moved inside parameters
        if "function_call" in _response:
            args = json.loads(_response["function_call"]["arguments"])
            return self._environment.get_template(template_name).render(**args)

    @classmethod
    def from_string_template(cls, template_string: str, llm_interface: LLMInterface,
                             **kwargs) -> "JinjaOpenAITemplateFunction":
        """
        This function takes a jinja2 template string and returns a function that can be used to render the template
        :param template_string: Jinja2 template string
        :param llm_interface: LLMInterface
        :param kwargs:  Additional arguments
        :return:
        """
        return cls(environment=Environment(
            loader=DictLoader({'template': template_string}), autoescape=True),
            llm_interface=llm_interface, **kwargs)

    @classmethod
    def from_environment(cls, environment: Environment, llm_interface: LLMInterface,
                         **kwargs) -> "JinjaOpenAITemplateFunction":
        """
        This function takes a jinja2 template string and returns a function that can be used to render the template
        :param environment: Jinja2 Environment
        :param llm_interface: LLMInterface
        :param kwargs: Additional arguments
        :return:
        """
        return cls(environment=environment, llm_interface=llm_interface, **kwargs)

    @classmethod
    def from_template_file(cls, template_file: str, llm_interface: LLMInterface,
                           **kwargs) -> "JinjaOpenAITemplateFunction":
        """
        This function takes a jinja2 template string and returns a function that can be used to render the template

        :param template_file: Path to the template file
        :param llm_interface: LLMInterface
        :param kwargs: Additional arguments
        :return:
        """
        with open(template_file) as f:
            template_string = f.read()
            _instance = cls(environment=Environment(
                loader=DictLoader({'template': template_string, template_file: template_string}), autoescape=True),
                llm_interface=llm_interface, **kwargs)
            return _instance

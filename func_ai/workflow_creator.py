import os

from func_ai.function_indexer import FunctionIndexer

curdir = os.path.dirname(os.path.realpath(__file__))


def create_workflow(steps: list):
    """
    Суздаване на workflow от стъпки

    :param steps: Лист от стъпки
    :return:
    """
    print(steps)
    # TODO create a system prompt that includes the steps and instructions to ask the user to provide any missing information

    return "ok"


def random_function(query: str) -> list:
    """
    Връща случйно число от 1 до 3

    :param query:  Заявка
    :return:
    """
    return "ok"


def main(query):
    _indexer = FunctionIndexer(f"{curdir}/function_indexer")
    _indexer.reset_function_index()
    _indexer.index_functions([])

    _catalog_str = "\n".join([f"- {fn}:{fd}" for fn, fd in _indexer.get_ai_fn_abbr_map().items()])
    # _system_prompt = f"""
    # Ти си помощник за планиране на действия. Твоята цел е от зададен системен каталог с операции да създадеш план за изпълнение на целите на потребителя.
    #
    # Системен Каталог с операции:
    # {_catalog_str}
    #
    # Пример:
    # - function_search_web_site: намери лекарство в даден сайт
    # - function_get_info: вземи информация за лекарство
    # - function_to_index_info: Индексирай информация за лекарство
    #
    #
    # Правила:
    # - Ще генерираш план за изпълнение на целите на потребителя под формата на лист с операции.
    # - Ще използваш само функции от системния каталог.
    #
    #
    # Рестрикции:
    # - Не използвай функции, които не са в системния каталог.
    # - Не генерирай нищо друго освен лист с операции.
    # """
    _system_prompt = f"""
Роля: Ти си експерт в планирането на действия, който използва системен каталог с операции за създаване на план за изпълнение на целите на потребителя.

Задача: Използвай зададения системен каталог с операции, за да създадеш детайлен план за изпълнение на целите на потребителя. Този план трябва да бъде представен като списък с операции, които трябва да бъдат изпълнени.

Контекст: Системният каталог с операции включва следните функции:
{_catalog_str}

Примери за функции от системния каталог:
- function_search_web_site намери лекарство в даден сайт
- function_get_info: вземи информация за лекарство
- function_to_index_info: Индексирай информация за лекарство

Правила:
- Генерирай план за изпълнение на целите на потребителя под формата на списък с операции.
- Използвай само функции от системния каталог.
- Използвай само имената на фукнции от системния каталог.

Ограничения:
- Не използвай функции, които не са в системния каталог.
- Не генерирай нищо друго освен списък с операции без допълнителни коментари или инфорамция.
- Не добавяй параметри на фунциите от системния каталог.

Когато свършиш с планирането спри и не генерирай повече съобщения.
"""

    _query = f"""
    {query}
    """

    _messages = [{
        "role": "system",
        "content": _system_prompt
    },
        {
            "role": "user",
            "content": _query
        }]
    _ai_fun_map, _fns_map, _fns_index_map = FunctionIndexer.get_functions([create_workflow])
    _messages_1 = run_function_loop(_messages, None, None)
    print("Done")
    # The bellow message is useful to reduce the output token count and prevent the model from generating unneccecary messages

    _new_system_prompt = f"""
    Създай воркфлоу от следните стъпки:
    {_messages_1[-1]["content"]}
    """
    _new_messages = [{
        "role": "user",
        "content": _new_system_prompt
    }, ]
    _new_messages.append(
        {"role": "assistant",
         "content": "After executing the given function  I will stop and will not generate any more messages"})
    _messages_2 = run_function_loop(_new_messages, _ai_fun_map, _fns_map)
    print(_messages_2)
    calculate_costs()
    reset_usage()


if __name__ == '__main__':
    main("Искам да добавя Панадол Бебе в базата данни.")

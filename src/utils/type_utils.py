from typing import Any, Type


def check_list_type(obj: list[Any], element_type: Type):
    return bool(obj) and all(isinstance(elem, element_type) for elem in obj)

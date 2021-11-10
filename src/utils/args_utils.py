import sys
import argparse
import dataclasses
from typing import Type, TypeVar, Generic

ContainerType = TypeVar('ContainerType')

"""
Custom Args Parser. Implementation was derived from the following source:
https://gist.github.com/dmitriy-serdyuk/bd434258e90697bafc5969a06083af73
"""


class Arg:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class Int(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=int, **kwargs)


class Float(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=float, **kwargs)


class Str(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=str, **kwargs)


class _MetaChoice(type):
    def __getitem__(self, item):
        return self(choices=list(item), type=item)


class Choice(Arg, metaclass=_MetaChoice):
    def __init__(self, choices, **kwargs):
        super().__init__(choices=choices, **kwargs)


class DataclassArgumentParser(Generic[ContainerType]):

    @staticmethod
    def mangle_name(name):
        return '--' + name.replace('_', '-')

    @staticmethod
    def initialize_parser(parser, container_class):
        for field in dataclasses.fields(container_class):
            name = field.name
            default = field.default
            value_or_class = field.type
            if isinstance(value_or_class, type):
                value = value_or_class(default=default)
            else:
                value = value_or_class
                value.kwargs['default'] = default
            parser.add_argument(DataclassArgumentParser.mangle_name(name), **value.kwargs)

        return parser

    def __init__(self, container_class: Type[ContainerType], parser: argparse.ArgumentParser = None):
        self.parser = argparse.ArgumentParser(description=container_class.__doc__) if parser is None else parser
        self.container_class = container_class
        self.initialize_parser(self.parser, self.container_class)

    def parse_args(self, **kwargs) -> ContainerType:
        if kwargs is None:
            kwargs = sys.argv[1:]

        arg_dict = self.parser.parse_args(**kwargs)
        return self.container_class(**vars(arg_dict))

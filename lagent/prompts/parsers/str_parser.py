from typing import Any

from lagent.registry import PARSER_REGISTRY, AutoRegister


class StrParser(metaclass=AutoRegister(PARSER_REGISTRY)):

    def __init__(
        self,
        template: str = '',
        **format_field,
    ):

        self.template = template
        self.format_field = format_field

    def format(self) -> str:
        format_data = {
            key: self.format_to_string(value)
            for key, value in self.format_field.items()
        }
        return self.template.format(**format_data)

    def format_to_string(self, format_model: Any) -> str:
        return format_model

    def parse(self, data: str) -> str:
        return data
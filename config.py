import json

from typing import Dict, Any


class TrainConfig:
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, TrainConfig(value))
            elif isinstance(value, list):
                setattr(self, key, [TrainConfig(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)

        return cls(data)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

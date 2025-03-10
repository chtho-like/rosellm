import argparse
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    seed: int = 42
    logging_level: str = "DEBUG"


@dataclass
class ModelConfig:
    name: str = "gpt2"


@dataclass
class DatasetConfig:
    path: str = "DigitalLearningGmbH/MATH-lighteval"
    name: str = "default"


@dataclass
class GlobalConfig:
    training: TrainingConfig
    model: ModelConfig
    dataset: DatasetConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GlobalConfig":
        return cls(
            training=TrainingConfig(**config_dict["training"]),
            model=ModelConfig(**config_dict["model"]),
            dataset=DatasetConfig(**config_dict["dataset"]),
        )


class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.config_path}")
        return self.config

    def get_config(self):
        if self.config is None:
            raise ValueError("Config not loaded")
        return self.config


class Args:
    def __init__(self, args: argparse.Namespace):
        self.config = Config(args.config_path)
        cfg = self.config.load_config()
        self.global_config = GlobalConfig.from_dict(cfg)
        self._args = args
        self._parser = args._parser if hasattr(args, "_parser") else None
        self.set_parser_ref(self._parser)

    def __getattr__(self, name: str):
        if hasattr(self._args, name):
            return getattr(self._args, name)
        raise AttributeError(f"Attribute {name} not found")

    # Static parser reference for type hints.
    _parser_ref = None

    @classmethod
    def set_parser_ref(cls, parser):
        """Set the parser reference for type hints."""
        cls._parser_ref = parser

    def __class_getitem__(cls, name: str) -> Any:
        """To help IDEs understand the types of the attributes."""
        if cls._parser_ref and name in cls._parser_ref.arg_types:
            return cls._parser_ref.arg_types[name]
        return Any

    @property
    def training(self) -> TrainingConfig:
        return self.global_config.training

    @property
    def model(self) -> ModelConfig:
        return self.global_config.model

    @property
    def dataset(self) -> DatasetConfig:
        return self.global_config.dataset


class ArgumentParser(argparse.ArgumentParser):
    """
    The sole purpose of this class is to save the energy
    of the developer from writing type annotations.
    """

    def __init__(self, *args, **kwargs):
        # Initialize arg_types before calling super().__init__
        # because super().__init__ might call add_argument
        self.arg_types = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        name = action.dest
        arg_type = (
            kwargs.get("type", type(action.default))
            if action.default is not None
            else type(None)
        )
        self.arg_types[name] = arg_type
        return action


class Parser:
    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            "--config-path", type=str, default="configs/default.yaml"
        )

    def parse_args_and_config(self) -> Args:
        args = self.parser.parse_args()
        args._parser = self.parser
        return Args(args)


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args_and_config()
    print(args.config_path)
    print(args.training.seed)
    print(args.model.name)

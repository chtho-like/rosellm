import pytest

from rosellm.config import Parser


@pytest.fixture
def parser():
    return Parser()


def teset_parser_initialization(parser):
    assert parser is not None
    assert parser.parser is not None


def test_parse_args_and_config(parser, monkeypatch):
    monkeypatch.setattr("sys.argv", ["program"])
    args = parser.parse_args_and_config()
    assert args is not None

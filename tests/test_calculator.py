import pytest

from mini_agent import calculator


def test_basic_multiplication():
    assert calculator("12345 * 6789") == "83810205"


def test_parentheses_and_precedence():
    assert calculator("(10 + 2) * 3 - 5") == "31"


def test_division():
    assert calculator("100 / 4") == "25.0"


def test_negative_number():
    assert calculator("-15 + 7") == "-8"


def test_function_call_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported expression: Call",
    ):
        calculator("open(1)")


def test_variable_is_rejected():
    with pytest.raises(ValueError):
        calculator("x + 1")


def test_large_exponent_is_rejected():
    with pytest.raises(
        ValueError,
        match="Exponent is too large",
    ):
        calculator("2 ** 100")
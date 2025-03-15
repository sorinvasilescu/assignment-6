import pytest
from basic_functions import add, multiply


def test_add_positive_numbers():
    assert add(1, 2) == 3


def test_add_negative_numbers():
    assert add(-1, -2) == -3


def test_multiply_positive_numbers():
    assert multiply(1, 2) == 2


def test_multiply_negative_numbers():
    assert multiply(-1, -2) == 2

"""A failed model call is said in words a person can act on, never as the raw exception (2026-10-21 journeys).

    python -m pytest mlmi/tests/test_error_words.py -q
"""
from mlmi.backend.agent import error_words


def test_each_failure_says_what_to_do():
    assert "internet connection" in error_words("<urlopen error [WinError 10061] No connection could be made>")
    assert "did not accept the key" in error_words('401: {"error":{"message":"User not found.","code":401}}')
    assert "Settings" in error_words("401 Unauthorized")
    assert "busy" in error_words("429 Too Many Requests")
    assert "starting it again" in error_words("connection refused 127.0.0.1:8142")
    assert "problem answering" in error_words("HTTP 503 Service Unavailable")


def test_anything_else_is_one_short_line():
    assert error_words("odd\nsecond line") == "odd"
    assert error_words("") == "The model did not answer."


def test_a_local_failure_is_never_blamed_on_nebius():
    assert "this computer" in error_words("<urlopen error [WinError 10061] refused>", local=True)

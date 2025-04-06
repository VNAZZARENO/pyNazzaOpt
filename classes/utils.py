# utils.py
import itertools
import uuid

_id_counter = itertools.count(1)

def generate_unique_id() -> int:
    """
    Generates a unique sequential integer ID.
    """
    return next(_id_counter)

def generate_unique_uuid() -> str:
    """
    Generates a unique UUID4 hex string.
    """
    return uuid.uuid4().hex
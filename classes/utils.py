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

from datetime import datetime, timedelta
import pandas as pd

def get_third_friday(year, month):
    """Get the third Friday of a given month and year"""
    date = datetime(year, month, 1)
    days_until_friday = (4 - date.weekday()) % 7
    first_friday = date + timedelta(days=days_until_friday)
    third_friday = first_friday + timedelta(days=14)
    return pd.Timestamp(third_friday)
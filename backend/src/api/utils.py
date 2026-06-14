import math


def clean_value(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return clean_record(v)
    if isinstance(v, list):
        return [clean_value(item) for item in v]
    return v


def clean_record(record: dict) -> dict:
    return {k: clean_value(v) for k, v in record.items()}


def clean_records(records: list) -> list:
    return [clean_record(r) for r in records]

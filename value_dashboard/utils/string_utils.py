import re


def find_between(s: str, start: str, end: str) -> str:
    match = re.search(f"{re.escape(start)}(.*?){re.escape(end)}", s)
    return match.group(1) if match else ""


def strtobool(val: str | bool) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def capitalize(fields: list) -> list:
    """Applies automatic capitalization.
    Parameters
    ----------
    fields : list
        A list of names

    Returns
    -------
    fields : list
        The input list, but each value properly capitalized
    """
    capitalize_end_words = [
        "ID",
        "Key",
        "Name",
        "Treatment",
        "Count",
        "Category",
        "Class",
        "Time",
        "DateTime",
        "UpdateTime",
        "Version",
        "Rate",
        "Ratio",
        "Negatives",
        "Positives",
        "Threshold",
        "Error",
        "Importance",
        "Type",
        "Percentage",
        "Index",
        "Symbol",
        "ResponseCount",
        "ConfigurationName",
        "Configuration",
    ]
    if not isinstance(fields, list):
        fields = [fields]
    fields_new = [re.sub("^p([xyz])", "", field) for field in fields]
    seen = set(fields)
    for i, item in enumerate(fields_new):
        if item in seen:
            fields_new[i] = fields[i]
    for word in capitalize_end_words:
        fields_new = [re.sub(word + '\b', word, field, flags=re.I) for field in fields_new]
        fields_new = [field[:1].upper() + field[1:] for field in fields_new]
    return fields_new


def isBool(val):
    if isinstance(val, int):
        return (val == 1) or (val == 0)
    if not (isinstance(val, str)):
        return False
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return True
    else:
        return False

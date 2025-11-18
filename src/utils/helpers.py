import re


def clean_filename(name: str) -> str:
    """
    Clean a filename for safe storage.
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"(?u)[^-\w.]", "", name)
    return name

"""Pull the chosen letter out of a model reply."""

import re

# A choice letter that is not part of a longer Latin word ("Because", "ABCD").
_LETTER = re.compile(r"(?<![A-Za-z])[ABCD](?![A-Za-z])")


def extract_letter(text):
    """The single choice letter in a reply, or None when there is none or more than one."""
    found = set(_LETTER.findall(text))
    return found.pop() if len(found) == 1 else None

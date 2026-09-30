"""Conservative final-answer matching for the project's fixed local fixtures."""
import re


def boxed(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def normalize(value):
    if value is None:
        return None
    value = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", value)
    return re.sub(r"[\s,$]", "", value)


def is_correct(text, expected, finish):
    return finish == "stop" and normalize(boxed(text)) == normalize(expected)

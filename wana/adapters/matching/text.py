import re
import unicodedata


def words(text: str, ignore: tuple[str, ...] = ()) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    for template in ignore:
        text = text.replace(unicodedata.normalize("NFC", template).casefold(), " ")
    return re.findall(r"\w+", text)


def shingles(text: str, size: int, ignore: tuple[str, ...] = ()) -> set[str]:
    tokens = words(text, ignore)
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}

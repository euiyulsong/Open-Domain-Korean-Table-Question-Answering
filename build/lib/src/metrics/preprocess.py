import string
import re

# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_punc(text):
        REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9가-힣]")
        return REMOVE_CHAR_PATTERN.sub(" ", text).strip()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()

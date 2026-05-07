import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH_MOVIES = os.path.join(PROJECT_ROOT, "data", "movies.json")
DATA_PATH_STOP_WORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_movies() -> list[dict]:
    with open(DATA_PATH_MOVIES, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]: 
    with open(DATA_PATH_STOP_WORDS, "r") as f:
        return f.read().splitlines()

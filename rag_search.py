# rag_search.py
# Основные функции для поиска по играм Steam
# Автор: GLEBowski
# Дата: 2026-03-13

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Словарь жанров
genre_keywords = {
    "шутер": [
        "shooter", "fps", "cs:go", "counter-strike", "call of duty",
        "team fortress", "apex", "pubg", "battlefield", "doom", "quake"
    ],
    "стратегия": [
        "strategy", "rts", "tactics", "civilization", "total war",
        "age of empires", "xcom", "stellaris", "crusader kings"
    ],
    "гонки": [
        "racing", "rally", "wrc", "nascar", "f1", "flatout",
        "need for speed", "forza", "gran turismo"
    ]
}

def load_vector_db(path="./steam_vectordb"):
    """Загружает векторную базу данных"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectordb = Chroma(persist_directory=path, embedding_function=embeddings)
    print(f"✅ База загружена. Игр в базе: {vectordb._collection.count()}")
    return vectordb

def find_shooters(vectordb, min_positive=10000):
    """Ищет шутеры с минимальным количеством отзывов"""
    shooter_keywords = genre_keywords["шутер"]
    found_games = {}

    for keyword in shooter_keywords[:10]:
        docs = vectordb.similarity_search(keyword, k=10)
        for doc in docs:
            name = doc.metadata.get("name", "")
            pos = doc.metadata.get("positive", 0)
            if pos >= min_positive and name not in found_games:
                found_games[name] = {
                    "positive": pos,
                    "dev": doc.metadata.get("developer", "?"),
                    "ccu": doc.metadata.get("ccu", 0),
                    "price": doc.metadata.get("price", 0)
                }

    sorted_games = sorted(found_games.items(), 
                         key=lambda x: x[1]["positive"], 
                         reverse=True)
    return sorted_games

def smart_game_answer(vectordb, query, k=30, min_positive=10, free_only=False):
    """Умный поиск игр по запросу"""
    docs = vectordb.similarity_search(query, k=k)

    games = []
    seen = set()

    for doc in docs:
        name = doc.metadata.get("name", "")
        if name in seen:
            continue
        seen.add(name)

        price = doc.metadata.get("price", 9999)
        positive = doc.metadata.get("positive", 0)

        if free_only and price != 0:
            continue
        if positive < min_positive:
            continue

        games.append(doc)

    games.sort(key=lambda x: x.metadata.get("positive", 0), reverse=True)
    return games[:10]

# Пример использования
if __name__ == "__main__":
    db = load_vector_db()
    shooters = find_shooters(db, min_positive=10000)
    for i, (name, data) in enumerate(shooters[:5], 1):
        print(f"{i}. {name} — {data['positive']} отзывов")

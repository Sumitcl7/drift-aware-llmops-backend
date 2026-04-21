from api.embedding_logger import log_query

queries = [
    "Top football players",
    "Best NBA teams",
    "How many players in soccer",
    "History of the Olympics",
    "Who won FIFA world cup",
    "Rules of cricket",
    "Best basketball strategy",
    "Famous tennis players",
    "How scoring works in rugby",
    "Sports analytics techniques"
]

for q in queries:
    log_query(q)
    print("Inserted:", q)
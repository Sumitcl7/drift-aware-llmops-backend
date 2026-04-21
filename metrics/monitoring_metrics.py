from prometheus_client import Counter, Gauge

query_counter = Counter("query_count", "Total number of queries")

cheap_model_usage = Counter("cheap_model_usage", "Cheap model usage")
mid_model_usage = Counter("mid_model_usage", "Mid model usage")
expert_model_usage = Counter("expert_model_usage", "Expert model usage")

drift_score_gauge = Gauge("drift_score", "Current embedding drift score")
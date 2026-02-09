from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "vision_agent_requests_total",
    "Total number of requests to VisionRAG Agent"
)

REQUEST_LATENCY = Histogram(
    "vision_agent_request_latency_seconds",
    "Latency of VisionRAG Agent requests"
)

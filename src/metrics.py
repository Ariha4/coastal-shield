"""
Coastal Shield -- Prometheus metrics for the Streamlit advisory-memo dashboard.

Exposes an HTTP /metrics endpoint (on a separate port from the Streamlit UI)
that Prometheus scrapes on an interval. Grafana then reads from Prometheus
to visualize trends over time -- memo volume by zone, generation latency,
and PDF export failures.
"""

from __future__ import annotations

import threading

from prometheus_client import Counter, Histogram, start_http_server

MEMOS_GENERATED = Counter(
    "coastal_shield_memos_generated_total",
    "Total number of advisory memos generated, by zone and source",
    ["zone", "source"],
)

MEMO_GENERATION_SECONDS = Histogram(
    "coastal_shield_memo_generation_seconds",
    "Time taken to generate and export an advisory memo",
)

PDF_EXPORT_FAILURES = Counter(
    "coastal_shield_pdf_export_failures_total",
    "Total number of PDF export failures",
)

_server_started = False
_lock = threading.Lock()


def ensure_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server exactly once per process.

    Streamlit re-runs the whole script top-to-bottom on every user
    interaction, so this guards against trying to bind the same port
    more than once per session.
    """
    global _server_started
    with _lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True

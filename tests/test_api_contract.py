import os

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:4566")
os.environ.setdefault("MYSQL_HOST", "localhost")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/")

from src.api.main import app  # noqa: E402
from src.api.routes_ingest import router as ingest_router  # noqa: E402


def test_required_routes_are_registered():
    # Inspect the app and the included router explicitly. This remains stable
    # across FastAPI versions that represent included routers differently.
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    paths.update(route.path for route in ingest_router.routes if hasattr(route, "path"))
    assert "/health" in paths
    assert "/raw/" in paths
    assert "/staging/" in paths
    assert "/curated/" in paths
    assert "/stats" in paths
    assert "/ingest" in paths
    assert "/ingest_fast" in paths

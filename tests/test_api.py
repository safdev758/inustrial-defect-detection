import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import io

from src.api import create_app

app = create_app()
_client_cm = TestClient(app)


def setup_module(module):
    _client_cm.__enter__()
    module.client = _client_cm


def teardown_module(module):
    _client_cm.__exit__(None, None, None)


def img_bytes(triple):
    a = np.full((64, 64, 3), triple, np.uint8)
    a[20:44, 20:44] = (200, 200, 200)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_inspect_returns_valid_decision():
    r = client.post("/inspect", files={"file": ("a.png", img_bytes(128), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"ANOMALY", "OK"}
    assert isinstance(body["anomaly_score"], float)
    assert body["classifier"]["label"] in {"crack", "cut", "normal", "unseen_type"}


def test_inspect_rejects_garbage():
    r = client.post("/inspect", files={"file": ("x.png", b"not-an-image", "image/png")})
    assert r.status_code == 400

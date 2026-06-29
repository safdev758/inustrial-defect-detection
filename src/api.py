import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from .models import ConvAutoencoder, build_classifier

CATEGORY = "hazelnut"
IMG_SIZE = 256
_CHECKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_METRICS = _CHECKPT_DIR.parent / "output" / "train" / "output" / "metrics.json"
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", _CHECKPT_DIR))
METRICS_PATH = Path(os.environ.get("METRICS_PATH", DEFAULT_METRICS))
SEEN_TYPES = ["crack", "cut"]


def load_image(f: UploadFile) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    img = Image.open(f.file).convert("RGB")
    return t(img)


class InspectResponse(BaseModel):
    decision: str
    anomaly_score: float
    threshold: float
    margin: float
    classifier: dict


def create_app():
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ae = ConvAutoencoder().to(device)
        ae.load_state_dict(torch.load(CHECKPOINT_DIR / f"autoencoder_{CATEGORY}.pth", map_location=device))
        ae.eval()
        clf = build_classifier(device)
        clf.load_state_dict(torch.load(CHECKPOINT_DIR / f"classifier_{CATEGORY}.pth", map_location=device))
        clf.eval()
        threshold = None
        if METRICS_PATH.exists():
            m = json.loads(METRICS_PATH.read_text())
            threshold = m["autoencoder"][CATEGORY]["best_threshold"]
        state.update(device=device, ae=ae, clf=clf, threshold=threshold)
        yield

    app = FastAPI(title="Industrial Defect Inspection API", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "category": CATEGORY, "threshold": state.get("threshold")}

    @app.post("/inspect", response_model=InspectResponse)
    def inspect(file: UploadFile = File(...)):
        if state.get("threshold") is None:
            raise HTTPException(status_code=503, detail="No metrics.json — run the training kernel first and pull output/")
        try:
            tensor = load_image(file).unsqueeze(0).to(state["device"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        with torch.no_grad():
            recon = state["ae"](tensor)
            score = float((recon - tensor).abs().amax())
            if score >= state["threshold"]:
                probs = torch.softmax(state["clf"](tensor), dim=1)[0]
                defect = SEEN_TYPES[int(probs.argmax())] if probs.max() > 0.5 else "unseen_type"
                verdict = {"label": defect, "probs": {t: round(float(p), 4) for t, p in zip(SEEN_TYPES, probs)}}
            else:
                verdict = {"label": "normal", "probs": {}}
        margin = round(score - state["threshold"], 5)
        return InspectResponse(
            decision="ANOMALY" if margin >= 0 else "OK",
            anomaly_score=round(score, 5),
            threshold=round(state["threshold"], 5),
            margin=margin,
            classifier=verdict,
        )

    return app


app = create_app()

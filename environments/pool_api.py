from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from android_control import EmulatorPool
import logging
import base64
import cv2
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def encode_image_to_base64(arr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', arr)  # 或 .png
    return base64.b64encode(buffer).decode('utf-8')

pool = EmulatorPool()

# ---------- Pydantic ----------
class ResetResponse(BaseModel):
    rollout_id: str
    device_id: str
    message: str
    screenshot_b64: str | None = None

class StepRequest(BaseModel):
    rollout_id: str
    action: dict

class StepResponse(BaseModel):
    ok: bool
    message: str
    before_b64: str | None = None
    after_b64: str | None = None
    action: dict | None = None

class CloseRequest(BaseModel):
    rollout_id: str

class CloseResponse(BaseModel):
    ok: bool

# ---------- Exception hook ----------
@app.exception_handler(Exception)
async def global_exc(request: Request, exc: Exception):
    logging.exception("[FastAPI] %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
async def reset_env():
    """
    return:
        "rollout_id": rollout_id,
        "device_id": device_id,
        "message": "reset ok",
        "screenshot_b64": screenshot_b64,
    """
    ticket = await pool.reset()
    if ticket is None:
        raise HTTPException(503, "No emulator available")
    return ticket

@app.post("/step", response_model=StepResponse)
async def step_env(req: StepRequest):
    ok, out = await pool.step(req.rollout_id, req.action)
    if not ok:
        raise HTTPException(400, out.get("error", "step failed"))
    return StepResponse(
        ok=True,
        message="success",
        before_b64=out["before"],
        after_b64=out["after"],
        action=out["action"],
    )

@app.post("/close", response_model=CloseResponse)
async def close_env(req: CloseRequest):
    ok = await pool.close(req.rollout_id)
    if not ok:
        raise HTTPException(400, "Invalid rollout_id")
    return CloseResponse(ok=True)

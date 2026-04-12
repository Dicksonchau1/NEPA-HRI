"""
NEPA-HRI Affective Modulation Layer — REST API v1.0
Part of the NEPA (Neuromorphic Edge Processing Architecture) platform.

Endpoints:
    GET  /health
    POST /session/evaluate   — full trace evaluation (1 billed session)
    POST /session/start      — create stateful streaming session
    POST /session/frame      — push one frame into live session
    DELETE /session/{id}     — end and clear session

Billing unit: 1 session = 1 call to /session/start or /session/evaluate

Run locally:
    pip install fastapi uvicorn numpy
    NEPA_API_KEYS=npa_test_xxxx uvicorn api.nepa_hri_api:app --reload
"""

import os
import uuid
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="NEPA-HRI API",
    description=(
        "Stability envelope control layer for LLM agents. "
        "Send an interaction trace, receive a formal AML context block "
        "ready to inject into any LLM system prompt."
    ),
    version="1.0.0",
    contact={"name": "AuraSense", "url": "https://github.com/Dicksonchau1/NEPA-HRI"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────
# Set env var: NEPA_API_KEYS=key1,key2,key3
def _load_keys() -> set:
    raw = os.environ.get("NEPA_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}

def verify_key(x_api_key: str = Header(...)):
    if x_api_key not in _load_keys():
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ── Model Parameters (matching simulator) ─────────────────────
MU_DECAY = 0.08
BETA     = 0.45
KAPPA    = 1.8
WIN_VI   = 4

# ── Pydantic Models ────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """
    Full interaction trace evaluation.
    Send the entire E(t) trace at once.
    Returns AML context block + ready-to-use prompt_injection string.
    Billed as 1 session.
    """
    interaction_trace: list[float] = Field(
        ...,
        description="E(t) emotional episode values. Length >= 2. Range [0, 1].",
        example=[0.25, 0.31, 0.45, 0.58, 0.74, 0.69]
    )
    incidents: Optional[list[float]] = Field(
        None,
        description="Incident magnitudes per frame (0.0 = none). Same length as trace.",
        example=[0.0, 0.0, 0.75, 0.0, 0.0, 0.0]
    )
    kappa: Optional[float] = Field(
        1.8,
        description="Stability envelope sensitivity kappa. Default 1.8."
    )
    desired_manifold: Optional[list[float]] = Field(
        None,
        description="E_desired(t) per frame. Defaults to constant 0.35 if omitted."
    )


class FrameRequest(BaseModel):
    """
    Single frame push for stateful streaming sessions.
    Call /session/start first to get a session_id.
    """
    session_id: str = Field(..., description="Session ID from /session/start")
    E_t: float = Field(..., ge=0.0, le=1.0, description="Current emotional episode value E(t)")
    incident: Optional[float] = Field(0.0, description="Incident magnitude this frame")
    E_desired: Optional[float] = Field(0.35, description="Desired emotional state this frame")


class AMLContext(BaseModel):
    """
    Affective Modulation Layer context block.
    Inject prompt_injection directly into your LLM system prompt.
    """
    session_id: str
    frame: int
    metrics: dict
    envelope: dict
    recommended_action: str  # LOCAL_HANDLE | ESCALATE | DEESCALATE
    prompt_injection: str    # ready-to-use system prompt block


# ── Core Computation ───────────────────────────────────────────

def _compute_aml(
    E: np.ndarray,
    incidents: np.ndarray,
    E_des: np.ndarray,
    kappa: float
) -> dict:
    """Compute full AML metrics from an E(t) trace."""
    T = len(E)
    S   = np.zeros(T)
    delta = np.zeros(T)
    F   = np.zeros(T)
    VI  = np.zeros(T)

    S[0] = 0.85
    F[0] = float(E[0])

    for t in range(1, T):
        # Neutral regulation assumption for batch trace eval
        regulation = 0.15
        S[t] = float(np.clip(
            S[t-1] * np.exp(-MU_DECAY) + BETA * regulation,
            0.02, 1.0
        ))
        delta[t] = float(E[t] - E_des[t])
        F[t] = float(0.75 * F[t-1] + 0.25 * E[t])
        w = E[max(0, t - WIN_VI): t + 1]
        VI[t] = float(np.var(w)) if len(w) > 1 else 0.0

    t = T - 1
    inside = bool(abs(delta[t]) <= kappa * S[t])
    AS_t   = float(F[t] - F[t-1]) if T > 1 else 0.0

    escalations = [i for i in range(1, T) if abs(delta[i]) > kappa * S[i]]
    p_esc = len(escalations) / (T - 1) if T > 1 else 0.0

    # Recover time
    recoveries, i = [], 0
    while i < len(escalations):
        esc_t, k = escalations[i], escalations[i] + 1
        while k < T and abs(delta[k]) > kappa * S[k]:
            k += 1
        if k < T:
            recoveries.append(k - esc_t)
        while i < len(escalations) and escalations[i] < k:
            i += 1
    t_recover = float(np.mean(recoveries)) if recoveries else 0.0

    if inside:
        action = "LOCAL_HANDLE"
    elif delta[t] > 0:
        action = "ESCALATE"
    else:
        action = "DEESCALATE"

    return {
        "t":                   t,
        "E_t":                 round(float(E[t]),    4),
        "S_t":                 round(float(S[t]),    4),
        "delta_t":             round(float(delta[t]),4),
        "volatility_index":    round(float(VI[t]),   4),
        "feeling_load":        round(float(F[t]),    4),
        "augmentation_score": round(AS_t,            4),
        "p_escalation":        round(p_esc,           4),
        "t_recover":           round(t_recover,       2),
        "inside_omega":        inside,
        "regime":              "A_stable" if inside else "B_escalation",
        "recommended_action":  action,
        "kappa":               kappa,
    }


def _build_prompt_block(r: dict) -> str:
    """Build ready-to-inject LLM system prompt block from AML result."""
    return (
        f"[NEPA-HRI Stability Context]\n"
        f"Regime: {r['regime']}\n"
        f"Stability Index S(t): {r['S_t']}\n"
        f"Affective Prediction Error delta(t): {r['delta_t']}\n"
        f"Volatility Index VI(t): {r['volatility_index']}\n"
        f"Feeling Load F(t): {r['feeling_load']}\n"
        f"Augmentation Score AS(t): {r['augmentation_score']}\n"
        f"P(escalation): {r['p_escalation']}\n"
        f"T_recover: {r['t_recover']} frames\n"
        f"Inside Omega: {r['inside_omega']}\n"
        f"Recommended Action: {r['recommended_action']}\n"
        f"\n"
        f"Rules:\n"
        f"- A_stable: respond normally, maintain engagement.\n"
        f"- B_escalation + ESCALATE: de-escalate, be concise, flag human operator.\n"
        f"- B_escalation + DEESCALATE: re-engage gently, reduce complexity.\n"
        f"- Never label user emotions directly.\n"
        f"- Never override an ESCALATE signal — always flag human operator."
    )


# ── In-memory session store (swap Redis in prod) ───────────────
_sessions: dict = {}


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Health check."""
    return {"status": "ok", "version": "1.0.0", "platform": "NEPA-HRI"}


@app.post(
    "/session/evaluate",
    response_model=AMLContext,
    tags=["Evaluation"],
    dependencies=[Depends(verify_key)],
    summary="Full trace evaluation (1 billed session)"
)
def evaluate(req: EvaluateRequest):
    """
    Send a complete interaction trace.
    Returns the AML context block + prompt_injection string.

    Billing: 1 session per call.
    """
    E = np.array(req.interaction_trace, dtype=float)
    if len(E) < 2:
        raise HTTPException(400, "interaction_trace must have >= 2 frames.")
    if np.any((E < 0) | (E > 1)):
        raise HTTPException(400, "All E(t) values must be in range [0, 1].")

    incidents = (
        np.array(req.incidents, dtype=float)
        if req.incidents
        else np.zeros(len(E))
    )
    E_des = (
        np.array(req.desired_manifold, dtype=float)
        if req.desired_manifold
        else np.full(len(E), 0.35)
    )
    kappa = req.kappa if req.kappa else KAPPA

    result = _compute_aml(E, incidents, E_des, kappa)
    sid = str(uuid.uuid4())

    return AMLContext(
        session_id=sid,
        frame=result["t"],
        metrics={
            "E_t":                result["E_t"],
            "S_t":                result["S_t"],
            "delta_t":            result["delta_t"],
            "volatility_index":   result["volatility_index"],
            "feeling_load":       result["feeling_load"],
            "augmentation_score": result["augmentation_score"],
            "p_escalation":       result["p_escalation"],
            "t_recover":          result["t_recover"],
        },
        envelope={
            "kappa":        result["kappa"],
            "inside_omega": result["inside_omega"],
            "regime":       result["regime"],
        },
        recommended_action=result["recommended_action"],
        prompt_injection=_build_prompt_block(result),
    )


@app.post(
    "/session/start",
    tags=["Streaming"],
    dependencies=[Depends(verify_key)],
    summary="Start a stateful streaming session (1 billed session)"
)
def start_session():
    """
    Create a new stateful session for real-time frame-by-frame evaluation.
    Returns a session_id. Use with /session/frame.

    Billing: 1 session per /session/start call.
    """
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "E":       [],
        "t":       0,
        "created": time.time(),
    }
    return {"session_id": sid, "status": "started"}


@app.post(
    "/session/frame",
    response_model=AMLContext,
    tags=["Streaming"],
    dependencies=[Depends(verify_key)],
    summary="Push one frame into a live session (no extra billing)"
)
def push_frame(req: FrameRequest):
    """
    Push a single E(t) observation into a live streaming session.
    Call this once per interaction turn in your agent loop.

    Billing: included in session — no extra charge per frame.
    """
    if req.session_id not in _sessions:
        raise HTTPException(
            404,
            "Session not found. Call /session/start first."
        )

    sess = _sessions[req.session_id]
    sess["E"].append(float(req.E_t))
    sess["t"] += 1

    E = np.array(sess["E"], dtype=float)
    E_des = np.full(len(E), req.E_desired if req.E_desired is not None else 0.35)
    incidents = np.zeros(len(E))
    if req.incident:
        incidents[-1] = req.incident

    result = _compute_aml(E, incidents, E_des, KAPPA)

    return AMLContext(
        session_id=req.session_id,
        frame=sess["t"],
        metrics={
            "E_t":                result["E_t"],
            "S_t":                result["S_t"],
            "delta_t":            result["delta_t"],
            "volatility_index":   result["volatility_index"],
            "feeling_load":       result["feeling_load"],
            "augmentation_score": result["augmentation_score"],
            "p_escalation":       result["p_escalation"],
            "t_recover":          result["t_recover"],
        },
        envelope={
            "kappa":        KAPPA,
            "inside_omega": result["inside_omega"],
            "regime":       result["regime"],
        },
        recommended_action=result["recommended_action"],
        prompt_injection=_build_prompt_block(result),
    )


@app.delete(
    "/session/{session_id}",
    tags=["Streaming"],
    dependencies=[Depends(verify_key)],
    summary="End and clear a session"
)
def end_session(session_id: str):
    """End a streaming session and free memory."""
    _sessions.pop(session_id, None)
    return {"status": "closed", "session_id": session_id}

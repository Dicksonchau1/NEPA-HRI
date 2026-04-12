"""Vercel serverless entrypoint for NEPA-HRI API."""
from api.nepa_hri_api import app  # noqa: F401 — Vercel picks up `app`

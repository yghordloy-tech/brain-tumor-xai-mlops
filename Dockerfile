# ──────────────────────────────────────────────────
#  Dockerfile — Brain Tumor XAI MLOps API
# ──────────────────────────────────────────────────

# TODO: Implement multi-stage Dockerfile
# Stage 1 — Builder
#   - Install Python dependencies
#   - Copy source code
#
# Stage 2 — Runtime
#   - Copy only needed artifacts from builder
#   - Copy model checkpoint
#   - Expose port 8000
#   - CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

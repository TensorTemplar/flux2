"""Entrypoints for FLUX.2 container deployment.

This module provides three phases for container deployment:
- startup: Fast-fail validation of environment and model availability
- setup: Download missing models to HF cache
- server: FastAPI inference service
"""

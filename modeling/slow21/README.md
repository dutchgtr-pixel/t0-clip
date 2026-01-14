# Slow21 Model (Public)

This directory contains the public training script for the Slow21 tail-risk model.
The script is designed to train against T₀-certified feature-store entrypoints and
uses censoring-aware survival modeling with boundary-focused evaluation.

Configuration is via environment variables and CLI flags; no secrets are committed.

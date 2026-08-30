"""HTTP edge for GlowMark analyze: Flask routes that validate uploads and call inference."""

import os
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import analyze_image_bytes, AnalyzeError
from suggestion_summary import ai_summary_enabled, set_ai_summary_enabled


def _warmup_steps():
    """Every cached artifact the serve path needs, in load order.

    Imported lazily so a failure here is reported by /health rather than preventing
    the process from starting — a container that cannot answer the probe gives no
    clue why, whereas a 503 naming the component does.
    """
    import gates
    import inference
    import region
    import region_stats
    import suggestion_serve

    return (
        ("gate_config", gates.load_gate_config),
        ("face_landmarker", inference._face_landmarker),
        ("beauty_model", inference._load_models),
        ("clip_realness", gates._clip_bundle),
        ("region_model", region._load_region_model),
        ("region_reference_stats", region_stats.load_reference_stats),
        ("suggestion_ranker", lambda: suggestion_serve.predict_suggestions(
            {c: 0.5 for c in __import__("geometry").FEATURE_COLS}, top_k=1)),
    )


def create_app() -> Flask:
    """Factory: wire CORS + routes; returns the Flask app."""
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        """Readiness probe. Loads every model, so a startup probe warms them.

        Each component is lru_cached, so this is expensive once and free afterwards.
        Doing it here moves cold-start cost onto the probe instead of onto the first
        real user, and turns a missing or corrupt artifact into a failed revision
        rather than a 500 on someone's upload.
        """
        loaded, failed = [], {}
        for name, load in _warmup_steps():
            try:
                load()
                loaded.append(name)
            except Exception as e:  # noqa: BLE001 — report, do not raise
                failed[name] = str(e)[:200]

        body = {"ok": not failed, "loaded": loaded}
        if failed:
            body["failed"] = failed
        return jsonify(body), (200 if not failed else 503)

    @app.post("/analyze")
    def analyze():
        """Validate image upload (type/size) → analyze_image_bytes → map errors to JSON/HTTP."""
        if "image" not in request.files:
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        file = request.files["image"]
        if not file or not getattr(file, "mimetype", ""):
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        if file.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        # 5MB
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 5 * 1024 * 1024:
            return jsonify({"error": "FILE_TOO_LARGE"}), 400

        image_bytes = file.read()
        # Optional comparison-group override. Session-only: never stored, just used
        # for this request. An unrecognised value is ignored inside the pipeline.
        region_override = request.form.get("region_override") or None
        try:
            result: Dict[str, Any] = analyze_image_bytes(
                image_bytes=image_bytes,
                region_override=region_override,
            )
        except AnalyzeError as e:
            body: Dict[str, Any] = {"error": e.code, "details": e.details}
            # Specific, actionable text (e.g. "Please close your mouth"); the frontend
            # prefers it over the generic per-code message.
            if e.hint:
                body["hint"] = e.hint
            return jsonify(body), e.http_status
        except Exception as e:
            return jsonify({"error": "UNKNOWN_ERROR", "details": str(e)}), 500

        return jsonify(result)

    @app.get("/api/settings/summary")
    def get_summary_setting():
        """Read whether the AI summary is enabled. Drives the Settings toggle."""
        return jsonify({"use_llm": ai_summary_enabled()}), 200

    @app.post("/api/settings/summary")
    def set_summary_setting():
        """Persist the AI-summary switch. body: {"use_llm": bool}."""
        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("use_llm"))
        set_ai_summary_enabled(enabled)
        return jsonify({"use_llm": enabled}), 200

    return app


if __name__ == "__main__":
    # Dev server: PORT env or 5001 (avoids colliding with Next.js on 3000).
    port = int(os.environ.get("PORT", "5001"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)

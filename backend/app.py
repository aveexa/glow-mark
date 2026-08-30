"""HTTP edge for GlowMark analyze: Flask routes that validate uploads and call inference."""

import os
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import analyze_image_bytes, AnalyzeError


def create_app() -> Flask:
    """Factory: wire CORS + routes; returns the Flask app."""
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        """Liveness probe for ops / load balancers."""
        return jsonify({"ok": True})

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

    return app


if __name__ == "__main__":
    # Dev server: PORT env or 5001 (avoids colliding with Next.js on 3000).
    port = int(os.environ.get("PORT", "5001"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)

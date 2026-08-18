"""Flask HTTP surface for Cloud Run orchestrator (same contract as backend/app.py)."""

from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from analyze import AnalyzeError, analyze_image_bytes


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/analyze")
    def analyze():
        if "image" not in request.files:
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        file = request.files["image"]
        if not file or not getattr(file, "mimetype", ""):
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        if file.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return jsonify({"error": "INVALID_FILE_TYPE"}), 400

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > 5 * 1024 * 1024:
            return jsonify({"error": "FILE_TOO_LARGE"}), 400

        image_bytes = file.read()
        try:
            result: Dict[str, Any] = analyze_image_bytes(image_bytes=image_bytes)
        except AnalyzeError as e:
            return jsonify({"error": e.code, "details": e.details}), e.http_status
        except Exception as e:
            return jsonify({"error": "UNKNOWN_ERROR", "details": str(e)}), 500

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

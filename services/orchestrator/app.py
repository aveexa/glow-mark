"""Flask HTTP surface for Cloud Run orchestrator (same contract as backend/app.py).

SUPERSEDED — DO NOT DEPLOY. This is the v1 pipeline.

``backend/`` is the maintained serve path. This package is a second, complete copy
that still builds and runs: its requirements.txt pins ``mediapipe>=0.10.14,<0.10.30``,
which still has the ``mp.solutions`` API that was removed in 1.0.1, so nothing here
fails loudly. It will happily serve ``POST /analyze`` with none of the v2 work:

  * no realness gate      — cartoons, renders, statues and animals get scored
  * no neutrality gate    — expression displaces ~13 of the 24 features
  * no roll autocorrect   — and pose comes from the landmark heuristic, not the
                            FaceLandmarker transformation matrix
  * no region conditioning — one global reference population, fitted on 412 mostly
                            white faces, which is the thing v2 exists to replace
  * still calls the Feature MLP over HTTP, which backend/ removed in favour of exact
    region-conditioned p20/p80 cutoffs

Nothing in the repo routes to it: the frontend calls NEXT_PUBLIC_BACKEND_URL, which
defaults to backend/app.py on :5001 and is not set anywhere here. See the warning at
the top of docs/gcp_deploy_three_models.md before acting on that guide.
"""

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

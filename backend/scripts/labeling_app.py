#!/usr/bin/env python3
"""Local Flask UI for human refinement of suggestion labels.

Bind: 127.0.0.1:5055 only.

Usage (from repo root):

  python backend/scripts/labeling_app.py --packet data/labeling/packets/ann_a_primary.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SUBMISSION_FIELDS = [
    "sample_id",
    "annotator_id",
    "priority_order",
    "suggestion_ids",
    "agreement_flag",
    "notes",
    "labeled_at",
]

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Glow-Mark Labeling</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#9aa7b8; --accent:#3d8bfd; --ok:#3dd68c; --warn:#f0b429; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }
    header { padding:12px 20px; border-bottom:1px solid #243044; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
    header .meta { color:var(--muted); font-size:14px; }
    main { display:grid; grid-template-columns: 340px 1fr; gap:20px; padding:20px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
    .card { background:var(--card); border-radius:10px; padding:14px; }
    img.face { width:100%; max-height:420px; object-fit:contain; background:#000; border-radius:8px; }
    table.feat { width:100%; border-collapse:collapse; font-size:12px; }
    table.feat th, table.feat td { text-align:left; padding:4px 6px; border-bottom:1px solid #2a3548; }
    .y-low { color:#f07178; } .y-ok { color:var(--ok); } .y-high { color:var(--warn); }
    .cand { border:1px solid #2a3548; border-radius:8px; padding:10px; margin-bottom:8px; }
    .cand.dropped { opacity:0.4; }
    .cand .id { font-weight:600; font-family: ui-monospace, monospace; }
    .cand .txt { color:var(--muted); font-size:13px; margin-top:4px; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:8px; }
    button, select { background:#243044; color:var(--text); border:1px solid #334155; border-radius:6px; padding:8px 12px; cursor:pointer; }
    button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
    button:disabled { opacity:0.5; cursor:not-allowed; }
    .empty { padding:40px; text-align:center; color:var(--muted); }
    input[type=text] { background:#0f1419; border:1px solid #334155; color:var(--text); border-radius:6px; padding:8px; width:100%; }
  </style>
</head>
<body>
<header>
  <strong>Glow-Mark labeling</strong>
  <span class="meta">annotator={{ annotator_id }} · role={{ role }} · {{ done }}/{{ total }} done · remaining {{ remaining }}</span>
</header>
{% if not sample %}
  <div class="empty">Queue complete (or all samples already labeled). Export a new packet or merge submissions.</div>
{% else %}
<main>
  <section>
    <div class="card">
      <img class="face" src="{{ url_for('media', sample_id=sample.sample_id) }}" alt="{{ sample.sample_id }}"/>
      <p class="meta" style="margin:10px 0 0;">{{ sample.sample_id }} · split={{ sample.split }}</p>
    </div>
    <div class="card" style="margin-top:14px; max-height:360px; overflow:auto;">
      <strong>Features</strong>
      <table class="feat">
        <tr><th>feature</th><th>value</th><th>y</th></tr>
        {% for feat, val in sample.features.items() %}
        {% set y = sample.y_classes[feat] %}
        <tr>
          <td>{{ feat }}</td>
          <td>{{ '%.4f'|format(val) }}</td>
          <td class="y-{{ y }}">{{ y }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </section>
  <section class="card">
    <form method="post" action="{{ url_for('save') }}" id="label-form">
      <input type="hidden" name="sample_id" value="{{ sample.sample_id }}"/>
      <input type="hidden" name="index" value="{{ index }}"/>
      <input type="hidden" name="priority_order" id="priority_order" value=""/>
      <p><strong>Candidates</strong> (keep up to 4, reorder priority)</p>
      <div id="cands">
        {% for sid in sample.candidate_ids %}
        <div class="cand" data-sid="{{ sid }}">
          <label><input type="checkbox" class="keep" checked/> <span class="id">{{ sid }}</span></label>
          <div class="txt">{{ sample.catalog_texts.get(sid, '') }}</div>
          <div class="row">
            <button type="button" class="up">Up</button>
            <button type="button" class="down">Down</button>
          </div>
        </div>
        {% endfor %}
      </div>
      <div class="row" style="margin-top:12px;">
        <select id="add-id">
          <option value="">Add from catalog…</option>
          {% for sid in sample.all_catalog_ids %}
          <option value="{{ sid }}">{{ sid }}</option>
          {% endfor %}
        </select>
        <button type="button" id="add-btn">Add</button>
      </div>
      <div style="margin-top:12px;">
        <label class="meta">notes (optional)</label>
        <input type="text" name="notes" placeholder="optional note"/>
      </div>
      <div class="row" style="margin-top:16px;">
        <button type="submit" class="primary" id="save-btn">Save (≤4)</button>
        <a href="{{ url_for('index', i=index+1) }}"><button type="button">Skip</button></a>
        {% if index > 0 %}
        <a href="{{ url_for('index', i=index-1) }}"><button type="button">Back</button></a>
        {% endif %}
      </div>
      <p class="meta" id="count-msg" style="margin-top:10px;"></p>
    </form>
  </section>
</main>
<script>
const texts = {{ sample.all_catalog_texts | tojson }};
function selectedIds() {
  return [...document.querySelectorAll('#cands .cand')].filter(c => c.querySelector('.keep').checked).map(c => c.dataset.sid);
}
function updateCount() {
  const n = selectedIds().length;
  const msg = document.getElementById('count-msg');
  msg.textContent = n + ' selected (need 1–4)';
  document.getElementById('save-btn').disabled = n < 1 || n > 4;
  document.querySelectorAll('#cands .cand').forEach(c => {
    c.classList.toggle('dropped', !c.querySelector('.keep').checked);
  });
}
document.getElementById('cands').addEventListener('click', (e) => {
  const cand = e.target.closest('.cand');
  if (!cand) return;
  if (e.target.classList.contains('up')) {
    const prev = cand.previousElementSibling;
    if (prev) cand.parentNode.insertBefore(cand, prev);
  }
  if (e.target.classList.contains('down')) {
    const next = cand.nextElementSibling;
    if (next) cand.parentNode.insertBefore(next, cand);
  }
});
document.getElementById('cands').addEventListener('change', updateCount);
document.getElementById('add-btn').addEventListener('click', () => {
  const sid = document.getElementById('add-id').value;
  if (!sid) return;
  if ([...document.querySelectorAll('#cands .cand')].some(c => c.dataset.sid === sid)) return;
  const div = document.createElement('div');
  div.className = 'cand';
  div.dataset.sid = sid;
  div.innerHTML = `<label><input type="checkbox" class="keep" checked/> <span class="id">${sid}</span></label>
    <div class="txt">${(texts[sid]||'').replace(/</g,'&lt;')}</div>
    <div class="row"><button type="button" class="up">Up</button><button type="button" class="down">Down</button></div>`;
  document.getElementById('cands').appendChild(div);
  updateCount();
});
document.getElementById('label-form').addEventListener('submit', (e) => {
  const ids = selectedIds();
  if (ids.length < 1 || ids.length > 4) { e.preventDefault(); return; }
  document.getElementById('priority_order').value = ids.join('|');
});
updateCount();
</script>
{% endif %}
</body>
</html>
"""


def _load_packet(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _labeled_ids(submission_csv: Path) -> set[str]:
    if not submission_csv.is_file():
        return set()
    with submission_csv.open(newline="", encoding="utf-8") as f:
        return {r["sample_id"] for r in csv.DictReader(f) if r.get("sample_id")}


def _append_submission(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def create_app(packet_path: Path, images_dir: Path, submissions_dir: Path) -> Flask:
    packet = _load_packet(packet_path)
    if not packet:
        raise SystemExit(f"Empty packet: {packet_path}")

    annotator_id = packet[0].get("annotator_id") or "ann_unknown"
    role = packet[0].get("role") or "primary"
    submission_csv = submissions_dir / f"{annotator_id}.csv"

    app = Flask(__name__)
    app.config["PACKET"] = packet
    app.config["IMAGES_DIR"] = images_dir
    app.config["SUBMISSION_CSV"] = submission_csv
    app.config["ANNOTATOR_ID"] = annotator_id
    app.config["ROLE"] = role

    @app.get("/")
    def index():
        labeled = _labeled_ids(app.config["SUBMISSION_CSV"])
        packet_rows: list[dict] = app.config["PACKET"]
        # Find next unlabeled starting at ?i=
        start = int(request.args.get("i", 0))
        idx = None
        sample = None
        for j in range(start, len(packet_rows)):
            if packet_rows[j]["sample_id"] not in labeled:
                idx = j
                sample = packet_rows[j]
                break
        if sample is None and start > 0:
            # wrap search from 0
            for j in range(0, min(start, len(packet_rows))):
                if packet_rows[j]["sample_id"] not in labeled:
                    idx = j
                    sample = packet_rows[j]
                    break

        done = len(labeled & {r["sample_id"] for r in packet_rows})
        total = len(packet_rows)
        remaining = total - done
        return render_template_string(
            INDEX_HTML,
            sample=sample,
            index=idx if idx is not None else 0,
            annotator_id=app.config["ANNOTATOR_ID"],
            role=app.config["ROLE"],
            done=done,
            total=total,
            remaining=remaining,
        )

    @app.post("/save")
    def save():
        sample_id = request.form.get("sample_id", "").strip()
        priority = request.form.get("priority_order", "").strip()
        notes = request.form.get("notes", "").strip()
        index = int(request.form.get("index", 0))
        ids = [s for s in priority.split("|") if s]
        if not sample_id or not (1 <= len(ids) <= 4):
            abort(400, "Need 1–4 suggestion IDs")
        # validate IDs are in packet catalog for this sample if present
        packet_rows = app.config["PACKET"]
        sample = next((r for r in packet_rows if r["sample_id"] == sample_id), None)
        if sample is None:
            abort(400, "Unknown sample_id")
        allowed = set(sample.get("all_catalog_ids") or [])
        if allowed and any(i not in allowed for i in ids):
            abort(400, "ID not in active catalog")

        joined = "|".join(ids)
        _append_submission(
            app.config["SUBMISSION_CSV"],
            {
                "sample_id": sample_id,
                "annotator_id": app.config["ANNOTATOR_ID"],
                "priority_order": joined,
                "suggestion_ids": joined,
                "agreement_flag": app.config["ROLE"],
                "notes": notes,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return redirect(url_for("index", i=index + 1))

    @app.get("/media/<sample_id>")
    def media(sample_id: str):
        images_dir: Path = app.config["IMAGES_DIR"]
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = images_dir / f"{sample_id}{ext}"
            if path.is_file():
                return send_file(path)
        abort(404)

    @app.get("/api/progress")
    def progress():
        labeled = _labeled_ids(app.config["SUBMISSION_CSV"])
        packet_rows = app.config["PACKET"]
        done = len(labeled & {r["sample_id"] for r in packet_rows})
        return jsonify({"done": done, "total": len(packet_rows)})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Local labeling UI")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "images")
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labeling" / "submissions",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()

    if not args.packet.is_file():
        print(f"ERROR: packet not found: {args.packet}", file=sys.stderr)
        return 1

    app = create_app(args.packet, args.images_dir, args.submissions_dir)
    print(f"Labeling UI → http://{args.host}:{args.port}/  packet={args.packet}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

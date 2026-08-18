#!/usr/bin/env python3
"""Phase 5 evaluation report for the shipped suggestion ranker.

Reconfirms NDCG@4 / MAP@4, catalog coverage, confidence bins, spot-check sheet,
ship gate, and optional serve smoke (--smoke-analyze).

Usage (from repo root):

  python backend/scripts/report_suggestion_ranker_eval.py --smoke-analyze
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS  # noqa: E402
from suggestion_model import (  # noqa: E402
    decode_top_k,
    encode_dataset_row,
    load_checkpoint,
    parse_id_list,
)
from suggestion_rl import TOP_K, eval_ndcg_map, gold_relevance, ndcg_at_k  # noqa: E402
from suggestion_rules import load_catalog  # noqa: E402

BEAUTY_CKPT = REPO_ROOT / "backend" / "models" / "beauty_landmarks_best.pt"
FEATURE_CKPT = REPO_ROOT / "backend" / "models" / "feature_geometry_model.pt"
BCE_ROLLBACK = REPO_ROOT / "backend" / "models" / "suggestion_ranker_bce_v1.pt"


def _load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.astype(object).where(pd.notnull(df), "").to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def _priority_ids(row: dict[str, str]) -> list[str]:
    return parse_id_list(row.get("priority_order") or row.get("suggestion_ids") or "")


def _resolve_image_path(image_path: str) -> Path | None:
    p = Path(image_path)
    if p.is_file():
        return p
    cand = REPO_ROOT / image_path
    if cand.is_file():
        return cand
    return None


def _percentile(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return 0.0
    return float(np.percentile(xs, q))


@torch.no_grad()
def _logits_for_rows(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    vocab: list[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.stack([encode_dataset_row(r, feat_mu, feat_sd)[0] for r in rows]).astype(np.float32)
    rels = np.stack([gold_relevance(_priority_ids(r), vocab).numpy() for r in rows]).astype(np.float32)
    model.eval()
    logits = model(torch.from_numpy(xs)).cpu().numpy()
    return logits, rels


def _coverage_and_confidence(
    logits: np.ndarray,
    vocab: list[str],
    catalog: dict[str, dict[str, str]],
    k: int,
) -> tuple[dict[str, Any], list[str]]:
    warns: list[str] = []
    id_counts: Counter[str] = Counter()
    confs: list[float] = []
    illegal: list[str] = []
    vocab_set = set(vocab)
    catalog_set = set(catalog.keys())

    for i in range(logits.shape[0]):
        top = decode_top_k(logits[i], vocab, k=k)
        for sid, conf in top:
            id_counts[sid] += 1
            confs.append(float(conf))
            if sid not in vocab_set or sid not in catalog_set:
                illegal.append(sid)

    if illegal:
        raise RuntimeError(f"Predicted IDs outside catalog/vocab: {sorted(set(illegal))[:10]}")

    total_slots = max(sum(id_counts.values()), 1)
    unique = len(id_counts)
    top_share = (id_counts.most_common(1)[0][1] / total_slots) if id_counts else 0.0
    pct_vocab = unique / max(len(vocab), 1)

    if unique < 8:
        warns.append(f"coverage: unique predicted IDs={unique} < 8 (possible collapse)")
    if top_share > 0.40:
        warns.append(f"coverage: top-ID share={top_share:.3f} > 0.40 (possible collapse)")

    arr = np.asarray(confs, dtype=np.float64)
    hist_counts, _ = np.histogram(arr, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
    conf_stats = {
        "n_pred_slots": int(arr.size),
        "mean": round(float(arr.mean()) if arr.size else 0.0, 6),
        "p50": round(_percentile(arr, 50), 6),
        "p90": round(_percentile(arr, 90), 6),
        "histogram_bins": ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1]"],
        "histogram_counts": [int(c) for c in hist_counts.tolist()],
    }
    coverage = {
        "unique_pred_ids": unique,
        "vocab_k": len(vocab),
        "pct_vocab_covered": round(pct_vocab, 6),
        "top_id": id_counts.most_common(1)[0][0] if id_counts else "",
        "top_id_share": round(top_share, 6),
        "top15": id_counts.most_common(15),
        "all_ids_in_catalog": True,
    }
    return {"coverage": coverage, "confidence": conf_stats, "warns": warns}, warns


def _write_spotcheck(
    path: Path,
    rows: list[dict[str, str]],
    logits: np.ndarray,
    rels: np.ndarray,
    vocab: list[str],
    k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "image_path",
        "gold_priority_order",
        "pred_ids",
        "pred_confidences",
        f"row_ndcg@{k}",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows):
            top = decode_top_k(logits[i], vocab, k=k)
            pred_ids = [t[0] for t in top]
            pred_idx = [vocab.index(sid) for sid in pred_ids]
            w.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "image_path": row.get("image_path", ""),
                    "gold_priority_order": "|".join(_priority_ids(row)),
                    "pred_ids": "|".join(pred_ids),
                    "pred_confidences": "|".join(f"{c:.4f}" for _, c in top),
                    f"row_ndcg@{k}": f"{ndcg_at_k(pred_idx, torch.from_numpy(rels[i]), k=k):.6f}",
                }
            )


def _ship_gate(bakeoff: dict[str, Any], k: int = 4) -> dict[str, Any]:
    models = bakeoff.get("models") or {}
    bce = models.get("bce_v1") or {}
    rl = models.get("rl_v1") or {}
    bv = float(bce.get(f"val_ndcg@{k}", 0.0))
    bt = float(bce.get(f"test_ndcg@{k}", 0.0))
    rv = float(rl.get(f"val_ndcg@{k}", 0.0))
    rt = float(rl.get(f"test_ndcg@{k}", 0.0))
    val_ok = rv >= bv
    test_ok = rt >= (bt - 0.02)
    passed = val_ok and test_ok
    return {
        "rule": "rl.val_ndcg>=bce.val_ndcg AND rl.test_ndcg>=bce.test_ndcg-0.02",
        "bce_val_ndcg": bv,
        "bce_test_ndcg": bt,
        "rl_val_ndcg": rv,
        "rl_test_ndcg": rt,
        "val_ok": val_ok,
        "test_ok": test_ok,
        "ship_gate_pass": passed,
        "winner": "rl_v1" if passed else "bce_v1",
    }


def _assert_suggestions_contract(suggestions: list[dict[str, Any]], catalog: dict[str, dict[str, str]]) -> None:
    if not isinstance(suggestions, list):
        raise RuntimeError("suggestions must be a list")
    if len(suggestions) > 4:
        raise RuntimeError(f"suggestions length {len(suggestions)} > 4")
    for s in suggestions:
        for key in ("id", "text", "confidence"):
            if key not in s:
                raise RuntimeError(f"suggestion missing key {key}: {s}")
        if s["id"] not in catalog:
            raise RuntimeError(f"suggestion id not in catalog: {s['id']}")
        if not str(s["text"]).strip():
            raise RuntimeError(f"empty approved text for {s['id']}")


def _smoke_analyze(
    test_rows: list[dict[str, str]],
    catalog: dict[str, dict[str, str]],
) -> dict[str, Any]:
    import suggestion_serve

    suggestion_serve._load_ranker_bundle.cache_clear()

    smoke_row = None
    img_path: Path | None = None
    for row in test_rows:
        cand = _resolve_image_path(str(row.get("image_path") or ""))
        if cand is not None:
            smoke_row = row
            img_path = cand
            break
    if smoke_row is None or img_path is None:
        return {
            "analyze_status": "error_no_test_image",
            "detail": "No test row with resolvable image_path",
        }

    beauty_ok = BEAUTY_CKPT.is_file()
    feature_ok = FEATURE_CKPT.is_file()

    if beauty_ok and feature_ok:
        # Prefer real POST /analyze via Flask test client
        from app import create_app  # noqa: WPS433

        app = create_app()
        client = app.test_client()
        raw = img_path.read_bytes()
        mime = "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        resp = client.post(
            "/analyze",
            data={"image": (io.BytesIO(raw), img_path.name, mime)},
            content_type="multipart/form-data",
        )
        payload = resp.get_json(silent=True) or {}
        if resp.status_code != 200:
            return {
                "analyze_status": "error_analyze_http",
                "http_status": resp.status_code,
                "payload": payload,
                "sample_id": smoke_row.get("sample_id"),
                "image_path": str(img_path),
            }
        suggestions = payload.get("suggestions") or []
        _assert_suggestions_contract(suggestions, catalog)
        return {
            "analyze_status": "ok",
            "http_status": 200,
            "n_suggestions": len(suggestions),
            "suggestion_ids": [s["id"] for s in suggestions],
            "sample_id": smoke_row.get("sample_id"),
            "image_path": str(img_path),
        }

    # Fallback: suggestion path only (beauty/feature missing)
    feats = {c: float(smoke_row[c]) for c in FEATURE_COLS}
    suggestions = suggestion_serve.predict_suggestions(feats, top_k=4)
    _assert_suggestions_contract(suggestions, catalog)
    return {
        "analyze_status": "fallback_predict_suggestions_missing_beauty_feature",
        "beauty_ckpt_present": beauty_ok,
        "feature_ckpt_present": feature_ok,
        "n_suggestions": len(suggestions),
        "suggestion_ids": [s["id"] for s in suggestions],
        "sample_id": smoke_row.get("sample_id"),
        "image_path": str(img_path),
    }


def _render_md(phase5: dict[str, Any], spot_path: Path) -> str:
    cov = phase5["coverage"]
    conf = phase5["confidence"]
    ship = phase5["ship_gate"]
    smoke = phase5.get("serve_smoke") or {}
    warns = phase5.get("warns") or []
    lines = [
        "# Suggestion ranker — Phase 5 evaluation",
        "",
        f"**Evaluated at:** `{phase5['evaluated_at']}`  ",
        f"**Checkpoint:** `{phase5['ckpt']}`  ",
        "",
        "## Ranking metrics (greedy top-4)",
        "",
        f"| Split | NDCG@4 | MAP@4 | n |",
        f"|-------|--------|-------|---|",
        f"| val | {phase5['val_ndcg@4']:.4f} | {phase5['val_map@4']:.4f} | {phase5['n_val']} |",
        f"| test | {phase5['test_ndcg@4']:.4f} | {phase5['test_map@4']:.4f} | {phase5['n_test']} |",
        "",
        "## Ship gate (reaffirm vs bakeoff bce_v1 / rl_v1)",
        "",
        f"- Rule: `{ship['rule']}`",
        f"- Pass: **{ship['ship_gate_pass']}** (winner `{ship['winner']}`)",
        f"- BCE val/test NDCG@4: {ship['bce_val_ndcg']:.4f} / {ship['bce_test_ndcg']:.4f}",
        f"- RL  val/test NDCG@4: {ship['rl_val_ndcg']:.4f} / {ship['rl_test_ndcg']:.4f}",
        "",
        "## Catalog ID coverage (val ∪ test predictions)",
        "",
        f"- Unique predicted IDs: **{cov['unique_pred_ids']}** / vocab {cov['vocab_k']} "
        f"({cov['pct_vocab_covered']*100:.1f}%)",
        f"- Top ID: `{cov['top_id']}` share **{cov['top_id_share']:.3f}**",
        f"- All predicted IDs in catalog: {cov['all_ids_in_catalog']}",
        "",
        "## Confidence (sigmoid on greedy top-4 slots)",
        "",
        f"- mean={conf['mean']:.4f}  p50={conf['p50']:.4f}  p90={conf['p90']:.4f}  n={conf['n_pred_slots']}",
        f"- histogram bins: {conf['histogram_bins']}",
        f"- histogram counts: {conf['histogram_counts']}",
        "",
        "## Spot-check",
        "",
        f"- Sheet: `{spot_path}` ({phase5['spotcheck_n']} test rows, seed {phase5['spotcheck_seed']})",
        "",
        "## Serve smoke",
        "",
        f"- status: `{smoke.get('analyze_status', 'not_run')}`",
        f"- sample_id: `{smoke.get('sample_id', '')}`",
        f"- suggestion_ids: `{smoke.get('suggestion_ids', [])}`",
        "",
        "## Warns",
        "",
    ]
    if warns:
        lines.extend([f"- {w}" for w in warns])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 suggestion ranker evaluation report.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "backend" / "models" / "suggestion_ranker.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.parquet",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data" / "catalogs" / "suggestions.csv",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ranker_bakeoff_metrics.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_ranker_phase5_report.md",
    )
    parser.add_argument(
        "--spot-out",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "labeling"
        / "spotcheck"
        / "suggestion_ranker_spotcheck_25_test.csv",
    )
    parser.add_argument("--spot-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument(
        "--smoke-analyze",
        action="store_true",
        help="Attempt POST /analyze (fallback to predict_suggestions if beauty/feature missing).",
    )
    args = parser.parse_args()

    if not args.ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    if not args.dataset.is_file():
        csv_path = args.dataset.with_suffix(".csv")
        if csv_path.is_file():
            args.dataset = csv_path
        else:
            print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
            return 1

    catalog = load_catalog(args.catalog)
    bundle = load_checkpoint(args.ckpt)
    vocab = list(bundle["suggestion_ids"])
    feat_mu = bundle["feat_mu"]
    feat_sd = bundle["feat_sd"]
    model = bundle["model"]

    rows = _load_rows(args.dataset)
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not val_rows or not test_rows:
        print("ERROR: need non-empty val and test splits", file=sys.stderr)
        return 1

    logits_val, rel_val = _logits_for_rows(model, val_rows, vocab, feat_mu, feat_sd)
    logits_test, rel_test = _logits_for_rows(model, test_rows, vocab, feat_mu, feat_sd)
    val_ndcg, val_map = eval_ndcg_map(torch.from_numpy(logits_val), torch.from_numpy(rel_val), k=args.k)
    test_ndcg, test_map = eval_ndcg_map(torch.from_numpy(logits_test), torch.from_numpy(rel_test), k=args.k)

    logits_vt = np.concatenate([logits_val, logits_test], axis=0)
    sanity, warns = _coverage_and_confidence(logits_vt, vocab, catalog, k=args.k)

    rng = np.random.default_rng(args.seed)
    n_spot = min(args.spot_n, len(test_rows))
    spot_idx = sorted(rng.choice(len(test_rows), size=n_spot, replace=False).tolist())
    spot_rows = [test_rows[i] for i in spot_idx]
    spot_logits = logits_test[spot_idx]
    spot_rels = rel_test[spot_idx]
    _write_spotcheck(args.spot_out, spot_rows, spot_logits, spot_rels, vocab, k=args.k)

    # Load / merge bakeoff JSON
    bakeoff: dict[str, Any] = {"models": {}}
    if args.metrics_out.is_file():
        try:
            bakeoff = json.loads(args.metrics_out.read_text(encoding="utf-8"))
            if not isinstance(bakeoff, dict):
                bakeoff = {"models": {}}
            bakeoff.setdefault("models", {})
        except json.JSONDecodeError:
            bakeoff = {"models": {}}

    ship = _ship_gate(bakeoff, k=args.k)

    # If ship gate fails unexpectedly, restore BCE production
    restored = False
    if not ship["ship_gate_pass"] and BCE_ROLLBACK.is_file():
        import shutil

        shutil.copy2(BCE_ROLLBACK, args.ckpt)
        restored = True
        warns.append("ship_gate_failed: restored suggestion_ranker_bce_v1.pt → production")

    smoke: dict[str, Any] = {"analyze_status": "not_run"}
    if args.smoke_analyze:
        try:
            smoke = _smoke_analyze(test_rows, catalog)
        except Exception as e:  # noqa: BLE001 — record smoke failures
            smoke = {"analyze_status": "error_exception", "detail": str(e)}

    phase5: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(args.ckpt),
        "k": args.k,
        "vocab_k": len(vocab),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        f"val_ndcg@{args.k}": round(val_ndcg, 6),
        f"val_map@{args.k}": round(val_map, 6),
        f"test_ndcg@{args.k}": round(test_ndcg, 6),
        f"test_map@{args.k}": round(test_map, 6),
        "coverage": sanity["coverage"],
        "confidence": sanity["confidence"],
        "warns": warns,
        "spotcheck_n": n_spot,
        "spotcheck_seed": args.seed,
        "spotcheck_path": str(args.spot_out),
        "ship_gate": ship,
        "ship_gate_pass": ship["ship_gate_pass"],
        "production_restored_to_bce": restored,
        "serve_smoke": smoke,
    }

    bakeoff["phase5"] = phase5
    bakeoff["updated_at"] = datetime.now(timezone.utc).isoformat()
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(bakeoff, indent=2) + "\n", encoding="utf-8")

    md = _render_md(phase5, args.spot_out)
    args.report_md.write_text(md, encoding="utf-8")

    print(
        f"phase5  val_ndcg@{args.k}={val_ndcg:.4f}  test_ndcg@{args.k}={test_ndcg:.4f}  "
        f"unique_ids={sanity['coverage']['unique_pred_ids']}  "
        f"ship_gate_pass={ship['ship_gate_pass']}  "
        f"smoke={smoke.get('analyze_status')}"
    )
    for w in warns:
        print(f"WARN: {w}")
    print(f"Wrote {args.metrics_out}")
    print(f"Wrote {args.spot_out}")
    print(f"Wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""final_test.py — membuka data uji SATU KALI untuk konfigurasi final dan pembandingnya (Bagian 4 notebook).

Syarat: outputs_cp2/fixed_protocol/final_config.json sudah ditulis dan dikunci di Bagian 4 (dipilih dari validation),
dan outputs_cp2/fixed_protocol/TEST_OPENED belum ada. Setelah jalan, TEST_OPENED dibuat sehingga test tidak bisa dibuka lagi.

Jalankan dari folder CP2_Kelompok_5:  python final_test.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from srl_cp2 import config as C  # noqa: E402
from srl_cp2.store import stable_hash  # noqa: E402

FLAG = C.OUT / "TEST_OPENED"
CFG = C.OUT / "final_config.json"


def config_hash(cfg):
    return stable_hash({k: v for k, v in cfg.items() if k not in ("hash", "locked_at")})


def lock_config(cfg):
    """Dipanggil Bagian 4: tulis final_config.json yang terkunci."""
    cfg = dict(cfg, locked=True, locked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    cfg["hash"] = config_hash(cfg)
    CFG.write_text(json.dumps(cfg, indent=1))
    return cfg["hash"]


def check_lock():
    if not CFG.exists():
        return False, "final_config.json belum ada: pilih konfigurasi final di validation dulu (Bagian 4)."
    cfg = json.loads(CFG.read_text())
    if not cfg.get("locked"):
        return False, "final_config.json belum dikunci."
    if cfg.get("hash") != config_hash(cfg):
        return False, "final_config.json berubah setelah dikunci."
    if FLAG.exists():
        return False, f"Test sudah pernah dibuka ({FLAG.read_text()}). Hasilnya ada di {C.OUT / 'test_report.json'}."
    return True, "siap"


def main():
    ok, msg = check_lock()
    if not ok:
        raise SystemExit(msg)
    cfg = json.loads(CFG.read_text())
    FLAG.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | config {cfg['hash']}")

    import numpy as np
    from srl_cp2.data import event_set
    from srl_cp2.decoding import tweet_sets
    from srl_cp2.evaluation import by_slice, evaluate, tweet_metrics
    from srl_cp2.experiments import Data
    from srl_cp2.frames import frame_scores, to_frame
    from srl_cp2.inference import predict_entry

    from srl_cp2.evaluation import bootstrap_ci
    report = {"config_hash": cfg["hash"], "models": {}, "perbandingan": []}
    PRED = {}
    teks_of = lambda e: e.get("teks") or ("en" if e.get("lang") == "en" else "processed")
    ids0 = None
    for teks in sorted({teks_of(e) for e in cfg["models"]}):
        D = Data(teks=teks, include_test=True)
        test = D.test
        ids = [t["text_id"] for t in test]
        assert ids0 is None or ids == ids0, "urutan test harus sama di semua versi teks"
        ids0 = ids
        clean = set(json.loads((C.DATA_OUT / "test_bersih.json").read_text())["ids"])
        gold = [t["labels"] for t in test]
        pick = lambda P, keep: ([g for g, t in zip(gold, test) if keep(t)], [p for p, t in zip(P, test) if keep(t)])
        for entry in [e for e in cfg["models"] if teks_of(e) == teks]:
            pred, p_tw, _ = predict_entry(entry, test, cache="TEST")
            rep = {"semua": evaluate(gold, pred),
                   "test_bersih": evaluate(*pick(pred, lambda t: t["text_id"] in clean)),
                   "domain_indonesia": evaluate(*pick(pred, D.in_domain)),
                   "punya_kejadian": evaluate(*pick(pred, lambda t: bool(event_set(t["labels"])))),
                   "per_kejadian": by_slice(test, gold, pred, lambda t: C.EVENT_CODES[t["event"]]),
                   "per_bahasa": by_slice(test, gold, pred, lambda t: D.lang_map[t["text_id"]])}
            gold_f = [to_frame(t["tokens"], g) for t, g in zip(test, gold)]
            pred_f = [to_frame(t["tokens"], p, None if p_tw is None else q) for t, p, q in
                      zip(test, pred, p_tw if p_tw is not None else [None] * len(test))]
            rep["kartu_kejadian"] = frame_scores(gold_f, pred_f)
            if p_tw is not None:
                rep["tingkat_tweet"] = tweet_metrics([event_set(g) for g in gold], tweet_sets(p_tw))
            report["models"][entry["name"]] = rep
            (C.OUT / f"test_pred_{entry['name']}.json").write_text(json.dumps(
                {"ids": [t["text_id"] for t in test], "pred": pred,
                 "p_tw": None if p_tw is None else np.asarray(p_tw).round(4).tolist()}))
            print(f"{entry['name']}: macro-F1 tanpa O {rep['semua']['f1_macro_noO']:.4f} | F1 berbobot {rep['semua']['f1_w']:.4f}")
            PRED[entry["name"]] = pred
    # Sebelum vs sesudah di test: model final − setiap pembanding (seed sama), bootstrap per tweet. Label emas test sama
    # untuk versi processed dan raw (prediksi raw dipetakan ke token berlabel), jadi perbandingan lintas versi sah.
    by = {(e["group"], e["seed"]): e["name"] for e in cfg["models"] if teks_of(e) in ("processed", "raw")}
    subsets = {"semua": lambda t: True, "test_bersih": lambda t: t["text_id"] in clean, "domain_indonesia": D.in_domain}
    for (g, s), nm in by.items():
        if g != "final":
            continue
        for (g2, s2), nm2 in by.items():
            if g2 == "final" or s2 != s:
                continue
            for sub, keep in subsets.items():
                idx = [i for i, t in enumerate(test) if keep(t)]
                G = [gold[i] for i in idx]
                for key in ("f1_macro_noO", "f1_span", "f1_FALSE", "f1_real_events", "f1_w"):
                    d = bootstrap_ci(G, [PRED[nm][i] for i in idx], [PRED[nm2][i] for i in idx], key)
                    report["perbandingan"].append({"pembanding": g2, "seed": s, "subset": sub, "metrik": key, **d})
    (C.OUT / "test_report.json").write_text(json.dumps(report, indent=1, default=float))
    print("Laporan test:", C.OUT / "test_report.json")


if __name__ == "__main__":
    main()

"""Combined check for the college game-model changes kept under the keep rule, with the head-to-heads.

    python experiments/apply_check_cfb_game.py

Nov 4, 2026 (FEATURES.md "Re-judging past tests under the keep rule"). Re-judged as better on every
measure, each on its own: in the margin, coverage defenders missed last game (or its "change"
version), drive value (with or without finishing), a mid-season coaching change, and the unknown
quarterback at -0.35 EPA/dropback; in the total, returning offensive production (or overall returning
production) and line yards per rush. Here they are measured on today's model with the production
code (cfbmodel/extra.py, ratings.CFBRatingEngine(extra=...)), on the frame the Update college board
Action fits (every FBS-vs-FBS game, 2014 on; played games only are scored).

Measures (the tests'): the raw model's average miss and the blended projection's average miss, margin
or total as the change touches, on the tuning seasons (2017-2019) and the holdout (2021 on; 2020 left
out), paired by game with standard errors. Rule, written before the run:
  1. Each head-to-head (coverage: missed vs change; drive: dve vs dve + fin; returning production:
     offense vs overall) is scored with the piece added alone to today's model. The version better on
     every measure wins; if neither is, the version the re-judge listed first (missed, dve alone,
     offense) is taken.
  2. The margin set (chosen coverage + chosen drive + d_mid + QB -0.35) and the total set (chosen
     returning production + s_lyd) are each scored together against today's model. A set is kept
     only if better on every measure. If not, each piece is removed in turn; the piece whose removal
     makes every measure better than the full set is dropped (and named), and the rest re-checked.
  3. The closing-line spread and total records (market-adjusted sides with EV above 0, holdout
     return) are reported alongside, not used to decide.

Writes experiments/out/apply_check_cfb_game.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbmodel import config as C, data, extra, model  # noqa: E402
from cfbmodel.ratings import CFBRatingEngine  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out" / "apply_check_cfb_game.json"
# today's model (main, Nov 4, 2026)
MAIN_MARGIN = ["d_pass", "d_rush", "d_sr", "d_pts", "d_mkt", "d_qb", "hfa", "rest_diff", "d_talent_early", "d_rtprod", "d_qb_early", "d_talent"]
MAIN_TOTAL = ["s_pass", "s_rush", "s_sr", "s_pts", "s_mkt", "s_qb", "lg_pts", "indoor", "s_def_rtprod", "s_snaps", "wind_hi"]
H2H = {"coverage": ("d_miss_defcov", "d_chg_defcov"), "drive": (["d_dve"], ["d_dve", "d_fin"]), "returning offense": ("s_off_rtprod", "s_rtprod")}
QB_NEW = -0.35


def frames():
    games, tg, qb = data.load_all(refresh=False, verbose=False)
    games = games[games.season <= 2026]
    inputs = extra.load_inputs(games, verbose=True)
    rated = extra.rated_inputs(extra.load_plays(sorted(games.season.unique())), extra.load_drives(sorted(games.season.unique())), which=("lyd", "dve", "fin"))
    out = {}
    for name, qbv in (("qb_main", C.QB_REPLACEMENT), ("qb_new", QB_NEW)):
        saved = C.QB_REPLACEMENT
        C.QB_REPLACEMENT = CFBRatingEngine.QB_REPLACEMENT = qbv
        try:
            eng = CFBRatingEngine(tg, qb, games=games, extra=rated)
            f = model.build_frame(games, eng, inputs=inputs)
        finally:
            C.QB_REPLACEMENT = CFBRatingEngine.QB_REPLACEMENT = saved
        f["act_margin"], f["act_total"] = f.home_score - f.away_score, f.home_score + f.away_score
        out[name] = f
        print(f"  frame built ({name}: unknown QB {qbv})", flush=True)
    return games, out


def run(f, mf, tf):
    o, _ = model.walk_forward(f, margin_feats=mf, total_feats=tf)
    return o[o.played & o.model_margin.notna() & (o.season != 2020)].set_index("game_id")


def paired(base, new, col, actual):
    n = new.loc[base.index]
    e0, e1 = (base[col] - base[actual]).abs(), (n[col] - base[actual]).abs()
    res = {}
    for name, m in (("tune", base.season.between(2017, 2019)), ("hold", base.season >= C.HOLDOUT_FROM)):
        d = (e1 - e0)[m]
        res[name] = {"n": int(m.sum()), "base": round(float(e0[m].mean()), 4), "change": round(float(d.mean()), 4),
                     "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)}
    return res


def score(base, out, kind):
    cols = {"margin": (("model_margin", "act_margin"), ("proj_margin", "act_margin")), "total": (("model_total", "act_total"), ("proj_total", "act_total"))}[kind]
    r = {"raw": paired(base, out, *cols[0]), "blend": paired(base, out, *cols[1])}
    r["better_all"] = all(r[k][p]["change"] < 0 for k in ("raw", "blend") for p in ("tune", "hold"))
    return r


def fmt(r):
    return (f"raw {r['raw']['tune']['change']:+.4f} ± {r['raw']['tune']['se']:.4f} / {r['raw']['hold']['change']:+.4f} ± {r['raw']['hold']['se']:.4f}"
            f"   blend {r['blend']['tune']['change']:+.4f} ± {r['blend']['tune']['se']:.4f} / {r['blend']['hold']['change']:+.4f} ± {r['blend']['hold']['se']:.4f}"
            f"   {'BETTER ON ALL' if r['better_all'] else 'mixed'}")


def record(games, out):
    try:
        from cfbmodel import backtest
        graded, _ = backtest.grade_all(games, out.reset_index())
        res = {}
        for mk in ("Spread", "Total"):
            g = graded[(graded.source == "proj") & (graded.market == mk) & (graded.ev > 0) & (graded.season >= C.HOLDOUT_FROM)]
            res[mk] = {"n": int(len(g)), "roi": round(float(g.profit.mean()), 4) if len(g) else None}
        return res
    except Exception as exc:
        return {"error": str(exc)}


def main():
    games, F = frames()
    f = F["qb_main"]
    base = run(f, MAIN_MARGIN, MAIN_TOTAL)
    res = {"main_margin": MAIN_MARGIN, "main_total": MAIN_TOTAL, "h2h": {}, "pieces": {}}

    def one(name, mf=None, tf=None, frame=f):
        mf, tf = mf or MAIN_MARGIN, tf or MAIN_TOTAL
        kind = "total" if tf is not MAIN_TOTAL and mf is MAIN_MARGIN else "margin"
        r = score(base, run(frame, mf, tf), kind)
        print(f"{name:42s} {fmt(r)}", flush=True)
        return r

    # 1. head-to-heads, each piece alone
    chosen = {}
    for key, (a, b) in H2H.items():
        va, vb = (a if isinstance(a, list) else [a]), (b if isinstance(b, list) else [b])
        is_total = key == "returning offense"
        ra = one(f"{key}: {'+'.join(va)}", tf=MAIN_TOTAL + va) if is_total else one(f"{key}: {'+'.join(va)}", mf=MAIN_MARGIN + va)
        rb = one(f"{key}: {'+'.join(vb)}", tf=MAIN_TOTAL + vb) if is_total else one(f"{key}: {'+'.join(vb)}", mf=MAIN_MARGIN + vb)
        dom = lambda x, y: all(x[k][p]["change"] < y[k][p]["change"] for k in ("raw", "blend") for p in ("tune", "hold"))
        pick = vb if dom(rb, ra) and not dom(ra, rb) else va
        chosen[key] = pick
        res["h2h"][key] = {"+".join(va): ra, "+".join(vb): rb, "pick": pick}
        print(f"   -> {key}: {'+'.join(pick)}", flush=True)
    res["pieces"]["coaching change d_mid"] = one("coaching change: d_mid", mf=MAIN_MARGIN + ["d_mid"])
    res["pieces"]["unknown QB -0.35"] = one("unknown QB -0.35", frame=F["qb_new"])
    res["pieces"]["line yards s_lyd"] = one("line yards: s_lyd", tf=MAIN_TOTAL + ["s_lyd"])

    # 2. the sets together, with leave-one-out
    def set_check(kind, pieces, frame_for):
        names = list(pieces)
        def build(active):
            mf, tf, frame = list(MAIN_MARGIN), list(MAIN_TOTAL), f
            for n in active:
                feats, is_qb = pieces[n]
                if is_qb:
                    frame = F["qb_new"]
                elif kind == "margin":
                    mf = mf + feats
                else:
                    tf = tf + feats
            return mf, tf, frame
        mf, tf, frame = build(names)
        full = score(base, run(frame, mf, tf), kind)
        print(f"\n{kind.upper()} SET {names}\n   {fmt(full)}", flush=True)
        out = {"pieces": names, "full": full, "leave_one_out": {}, "dropped": []}
        while not full["better_all"] and len(names) > 1:
            cands = {}
            for skip in names:
                mf, tf, frame = build([n for n in names if n != skip])
                r = score(base, run(frame, mf, tf), kind)
                out["leave_one_out"][f"without {skip}"] = r
                print(f"   without {skip:30s} {fmt(r)}", flush=True)
                if all(r[k][p]["change"] < full[k][p]["change"] for k in ("raw", "blend") for p in ("tune", "hold")):
                    cands[skip] = sum(r[k][p]["change"] for k in ("raw", "blend") for p in ("tune", "hold"))
            if not cands:
                break
            drop = min(cands, key=cands.get)
            out["dropped"].append(drop)
            names = [n for n in names if n != drop]
            mf, tf, frame = build(names)
            full = score(base, run(frame, mf, tf), kind)
            print(f"   dropped {drop}; set {names}\n   {fmt(full)}", flush=True)
        out["kept"] = names if full["better_all"] else []
        out["final"] = full
        mf, tf, frame = build(out["kept"])
        out["margin_feats"], out["total_feats"], out["qb_new"] = mf, tf, frame is F["qb_new"]
        return out, run(frame, mf, tf)

    margin_pieces = {"coverage": (chosen["coverage"], False), "drive": (chosen["drive"], False), "coaching change": (["d_mid"], False), "unknown QB -0.35": ([], True)}
    total_pieces = {"returning offense": (chosen["returning offense"], False), "line yards": (["s_lyd"], False)}
    res["margin_set"], om = set_check("margin", margin_pieces, None)
    res["total_set"], ot = set_check("total", total_pieces, None)
    print(f"\nkept margin pieces: {res['margin_set']['kept']}  total pieces: {res['total_set']['kept']}", flush=True)
    res["records"] = {"main": record(games, base), "margin_set": record(games, om), "total_set": record(games, ot)}
    print("holdout records (EV>0): ", res["records"], flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""A quarterback's touchdown passes with a teammate's anytime touchdown: a nested model.

    python experiments/td_pairs.py

The same-game copula (nflmodel/propsim.py) gives every pair one correlation. For a
quarterback's passing touchdowns and his top receiver's anytime touchdown it runs short even
at its ceiling (29.0% predicted against 33.8% on the holdout), because the two are not merely
correlated: a receiving touchdown *is* one of the quarterback's touchdown passes. Once the
receiver has scored through the air, the quarterback is already over 0.5 and needs one fewer
for any higher line.

Nested model. For a quarterback over L passing touchdowns and a same-team receiver (R1-R3 by
target share, RB1) to score:

    P(both) = P(receiver scores) x q,   q = P(QB over L | the receiver scored)

with logit q = a + b x logit(P(QB over L)), one (a, b) per receiver role and line (0.5, 1.5,
2.5), fitted on the tuning seasons (2013-2018) where the receiver scored. It keeps each leg's
own chance exactly (the receiver's is the factor; the quarterback's enters through q).

Scored on the holdout (2019 on, chances from models fitted before 2019) against the copula with
the production correlations: predicted against actual share of both, Brier and log loss, by
role and line. Rule, set before running: it replaces the copula for a role and line when its
holdout Brier is lower; the fit goes to data/td_pairs.json for the Parlays page.

Writes experiments/out/td_pairs.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from nflmodel import config as C, propsim as S, props as P  # noqa: E402
from props_calibration import FAMILY, TUNE, load_proj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "out" / "td_pairs.json"
BOARD = C.ROOT / "data" / "td_pairs.json"
LINES = (0.5, 1.5, 2.5)
ROLES = ("R1", "R2", "R3", "RB1")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pairs(df: pd.DataFrame, model: P.PropModel) -> pd.DataFrame:
    """One row per (game, quarterback, teammate receiver, line): both legs' chances and outcomes."""
    d = df[df["played"].astype(bool) & (df["active"] == 1)].copy()
    d["role"] = S.roles(d)
    qb = d[d["role"] == "QB"].reset_index(drop=True)
    over = {}
    if len(qb):
        o, _, _ = model.probs(qb, "pass_td", "QB", np.tile(np.array(LINES), (len(qb), 1)))
        for j, L in enumerate(LINES):
            over[L] = o[:, j]
    q = qb[["game_id", "team", "season", "pass_td"]].copy()
    for L in LINES:
        q[f"p{L}"] = over[L]
    rec = []
    for role in ROLES:
        r = d[d["role"] == role]
        for grp in r["group"].unique():
            x = r[r["group"] == grp].reset_index(drop=True)
            if (("anytime_td", grp) not in model.counts) or x.empty:
                continue
            o, _, _ = model.probs(x, "anytime_td", grp, np.full(len(x), 0.5))
            rec.append(pd.DataFrame({"game_id": x["game_id"], "team": x["team"], "role": role, "pr": o,
                                     "scored": ((x["rec_td"].fillna(0) + x["rush_td"].fillna(0)) >= 1).astype(float),
                                     "rec_td": x["rec_td"].fillna(0)}))
    rec = pd.concat(rec, ignore_index=True)
    j = rec.merge(q, on=["game_id", "team"])
    rows = []
    for L in LINES:
        rows.append(j.assign(L=L, pq=j[f"p{L}"], qb_over=(j["pass_td"] > L).astype(float))[
            ["game_id", "season", "role", "L", "pr", "pq", "scored", "qb_over"]])
    return pd.concat(rows, ignore_index=True)


def fit(x: pd.DataFrame) -> dict:
    """(a, b) per role and line: logistic of qb_over on logit(pq), where the receiver scored."""
    out = {}
    for (role, L), g in x[x["scored"] == 1].groupby(["role", "L"]):
        z, y = logit(g["pq"].to_numpy()), g["qb_over"].to_numpy()
        a, b = 0.0, 1.0
        for _ in range(50):                                        # Newton steps
            p = 1 / (1 + np.exp(-(a + b * z)))
            w = p * (1 - p)
            H = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]]) + 1e-6 * np.eye(2)
            gr = np.array([(y - p).sum(), ((y - p) * z).sum()])
            step = np.linalg.solve(H, gr)
            a, b = a + step[0], b + step[1]
            if np.abs(step).max() < 1e-8:
                break
        out[f"{role}|{L}"] = [round(float(a), 4), round(float(b), 4), int(len(g))]
    return out


def nested(x: pd.DataFrame, ab: dict) -> np.ndarray:
    a = x.apply(lambda r: ab[f"{r.role}|{r.L}"][0], axis=1).to_numpy()
    b = x.apply(lambda r: ab[f"{r.role}|{r.L}"][1], axis=1).to_numpy()
    q = 1 / (1 + np.exp(-(a + b * logit(x["pq"].to_numpy()))))
    return x["pr"].to_numpy() * q


def copula(x: pd.DataFrame, rhos: dict) -> np.ndarray:
    rho = x["role"].map(lambda r: rhos.get(S.pair_key("team", "anytime_td", r, "pass_td", "QB"), [0.0])[0]).to_numpy(float)
    return S.joint_over(x["pr"].to_numpy(), x["pq"].to_numpy(), rho)


def score(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"said": round(float(p.mean()), 4), "brier": round(float(((p - y) ** 2).mean()), 5),
            "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 5)}


def main():
    proj = load_proj(None)
    tune = proj[proj.season.isin(TUNE)]
    m_t = P.PropModel(FAMILY).fit(tune)
    tr = pairs(tune, m_t)
    ab = fit(tr)
    ho = proj[proj.season >= C.HOLDOUT_FROM]
    m_h = P.PropModel(FAMILY).fit(proj[proj.season < C.HOLDOUT_FROM])
    saved = P.PropModel.load()
    if saved is not None:
        m_h.iso = saved.iso
    hx = pairs(ho, m_h)
    rhos = ((saved.meta or {}).get("correlation") or {}).get("pairs", {}) if saved else {}
    hx["both"] = hx["scored"] * hx["qb_over"]
    hx["p_cop"] = copula(hx, rhos)
    hx["p_nest"] = nested(hx, ab)
    hx["p_ind"] = hx["pr"] * hx["pq"]
    rows, use = [], {}
    for (role, L), g in hx.groupby(["role", "L"]):
        y = g["both"].to_numpy()
        r = {"role": role, "line": L, "n": int(len(g)), "actual": round(float(y.mean()), 4),
             "independent": score(g["p_ind"].to_numpy(), y), "copula": score(g["p_cop"].to_numpy(), y),
             "nested": score(g["p_nest"].to_numpy(), y)}
        r["nested_wins"] = r["nested"]["brier"] < r["copula"]["brier"]
        use[f"{role}|{L}"] = bool(r["nested_wins"])
        rows.append(r)
    # the production fit: every season through the last full one
    last_full = int(proj.loc[proj.played.astype(bool) & (proj.week >= 17), "season"].max())     # the last season played to the end
    full = proj[proj.season <= last_full]
    m_f = P.PropModel(FAMILY).fit(full)
    ab_full = fit(pairs(full, m_f))
    BOARD.write_text(json.dumps({"fitted_on": [int(full.season.min()), int(full.season.max())], "lines": list(LINES),
                                 "ab": ab_full, "use": use}, indent=1))
    OUT.write_text(json.dumps({"fit_tuning": ab, "holdout": rows}, indent=1))
    print(f"{'role':5s} {'L':>4s} {'n':>6s} {'actual':>7s} {'indep':>7s} {'copula':>7s} {'nested':>7s}   Brier cop / nest")
    for r in rows:
        print(f"{r['role']:5s} {r['line']:>4} {r['n']:>6d} {r['actual']:>7.3f} {r['independent']['said']:>7.3f} {r['copula']['said']:>7.3f} "
              f"{r['nested']['said']:>7.3f}   {r['copula']['brier']:.5f} / {r['nested']['brier']:.5f} {'*' if r['nested_wins'] else ''}")
    print(f"wrote {BOARD} and {OUT}")


if __name__ == "__main__":
    main()

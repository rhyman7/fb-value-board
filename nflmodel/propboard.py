"""The Props section of the NFL board: one file per week, built from the prop log.

  docs/props/<season>_w<WW>.json   every prop priced that week (main lines and ladder
                                   rungs), one record per player x market x line: the
                                   books' prices, our chances, the vig-free market chance,
                                   the edge, the closing line and the result
  docs/props/index.json            the weeks on file with a summary each, and the season

Rebuilt from data/props/ on every props run (run_props.py), so it always matches the log.
The page fetches these files when the Props section is opened (they are too big to inline).
What a week shows for a prop is its pricing snapshot (where bets are decided); before a
game has one, its latest opening snapshot, marked as early.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C, propdist as D, proplog as PLOG
from .propmarket import LADDER_GAP

OUT = C.DOCS / "props"
KEY = ["game_id", "player_id", "market", "line"]


def _week(game_id: str) -> int:
    return int(str(game_id).split("_")[1])


def _num(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), d)


def _clean(o):
    """JSON-safe: NaN and missing values become null (the page's JSON.parse rejects NaN)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NaT or (not isinstance(o, (str, bytes)) and pd.api.types.is_scalar(o) and pd.isna(o)):
        return None
    return o


def _dump(path, body):
    path.write_text(json.dumps(_clean(body), separators=(",", ":"), default=str, allow_nan=False))


def _books(s) -> dict:
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except ValueError:
        return {}


def records(log: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """One row per prop (both sides together) at its pricing snapshot, or latest open one."""
    if log.empty:
        return pd.DataFrame()
    log = log.copy()
    log["line"] = log["line"].astype(float)
    log["week"] = log["game_id"].map(_week)
    rank = {"flag": 2, "open": 1}
    base = log[log["snapshot"].isin(rank)].copy()
    base["rk"] = base["snapshot"].map(rank)
    # the pricing snapshot if there is one, else the latest opening snapshot
    base = base.sort_values(["rk", "time"]).groupby(KEY + ["side"], as_index=False).tail(1)
    best = base.groupby(KEY)["rk"].transform("max")
    base = base[base["rk"] == best]
    over = base[base["side"] == "over"].set_index(KEY)
    under = base[base["side"] == "under"].set_index(KEY)
    idx = over.index.union(under.index)
    o, u = over.reindex(idx), under.reindex(idx)
    first = o.combine_first(u)
    out = pd.DataFrame(index=idx)
    for c in ("season", "week", "player", "team", "pos", "stat", "proj", "books", "commence", "time", "snapshot",
              "n_books", "one_sided", "hold"):
        out[c] = first[c]
    out["p_over"] = o["our_p"]
    out["p_under"] = u["our_p"].where(u["our_p"].notna(), 1 - o["our_p"] - o["our_push"])
    out["p_push"] = first["our_push"]
    out["fair_over"] = o["fair"].where(o["fair"].notna(), 1 - u["fair"])
    out["edge_over"], out["edge_under"] = o["edge"], u["edge"]
    out["ev_over"], out["ev_under"] = o["ev"], u["ev"]
    out["price_over"], out["price_under"] = o["price"], u["price"]
    out["book_over"], out["book_under"] = o["book"], u["book"]
    out["flag_over"] = o["flagged"].fillna(0).astype(bool)
    out["flag_under"] = u["flagged"].fillna(0).astype(bool)
    out["stake_over"], out["stake_under"] = o["stake"], u["stake"]
    out["why_not"] = first["why_not"]
    out["row_over"], out["row_under"] = o["row_id"], u["row_id"]
    out = out.reset_index()
    # market-only tags (propmarket.market_tags), by the pricing row's id
    tg = PLOG.read_tags()
    if len(tg):
        tg = tg.drop_duplicates("row_id").set_index("row_id")
        to, tu = tg.reindex(out["row_over"]), tg.reindex(out["row_under"])
        out["book_gap"] = to["book_gap"].where(to["book_gap"].notna(), tu["book_gap"]).to_numpy()
        out["middle"] = to["middle"].where(to["middle"].notna(), tu["middle"]).fillna(0).to_numpy()
        out["ladder_gap"] = to["ladder_gap"].where(to["ladder_gap"].notna(), tu["ladder_gap"]).to_numpy()
        out["tag_book"] = to["tag_book"].where(to["tag_book"].notna(), tu["tag_book"]).to_numpy()
        out["take"] = np.where(to["take"].notna() & (to["take"] != ""), to["take"], np.where(tu["take"].notna(), tu["take"], "")).astype(object)
    else:
        out["book_gap"] = out["tag_book"] = out["take"] = None
        out["middle"], out["ladder_gap"] = 0, np.nan
    nw = PLOG.read_new()
    if len(nw):
        nw = nw.drop_duplicates("row_id").set_index("row_id")
        pno, pnu = nw["our_p_new"].reindex(out["row_over"]).to_numpy(), nw["our_p_new"].reindex(out["row_under"]).to_numpy()
        out["pn_over"] = pno
        out["proj_new"] = nw["proj_new"].reindex(out["row_over"]).to_numpy()
        out["pn_under"] = np.where(np.isnan(pnu), 1 - pno - nw["our_push_new"].reindex(out["row_over"]).fillna(0).to_numpy(), pnu)
    else:
        out["pn_over"] = out["pn_under"] = out["proj_new"] = np.nan

    # closing line for the same player and market: same line if the books still had it
    cl = log[log["snapshot"] == "close"]
    if len(cl):
        c_over = cl[cl["side"] == "over"].sort_values("time").groupby(KEY).tail(1)
        same = c_over.set_index(KEY)[["fair", "price"]].rename(columns={"fair": "close_fair_over", "price": "close_price_over"})
        out = out.join(same, on=KEY)
        main = cl[~cl["market"].str.endswith("_alternate")]
        cline = main.groupby(["game_id", "player_id", "market"])["line"].median().rename("close_line")
        out = out.join(cline, on=["game_id", "player_id", "market"])
    else:
        out["close_fair_over"] = out["close_price_over"] = out["close_line"] = np.nan

    # results (graded on the pricing rows)
    if len(res):
        r = res.set_index("row_id")
        for side in ("over", "under"):
            rr = r.reindex(out[f"row_{side}"])
            out[f"res_{side}"] = rr["result"].to_numpy()
            out[f"actual_{side}"] = rr["actual"].to_numpy()
        out["actual"] = out["actual_over"].where(out["actual_over"].notna(), out["actual_under"])
        out["result"] = np.where(out["res_over"].isin(["win", "loss", "push"]), out["res_over"],
                                 np.where(out["res_under"] == "win", "loss", np.where(out["res_under"] == "loss", "win",
                                          out["res_under"])))     # result of the over
        out.loc[out["result"].isin(["void"]), "actual"] = np.nan
    else:
        out["actual"], out["result"] = np.nan, None
    return out


def _move(r) -> float | None:
    """Points the main line moved toward the side we preferred (positive = toward us)."""
    if pd.isna(r["close_line"]) or str(r["market"]).endswith("_alternate"):
        return None
    side_over = (r["p_over"] or 0) / max((r["p_over"] or 0) + (r["p_under"] or 0), 1e-9) > (r["fair_over"] or 0.5)
    return float(r["close_line"] - r["line"]) * (1 if side_over else -1)


def summary(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"props": 0}
    main = x[~x["market"].str.endswith("_alternate")]
    dec = x[x["result"].isin(["win", "loss"]) & x["fair_over"].notna() & x["p_over"].notna()]
    hit = (dec["result"] == "win").astype(float)
    ours = dec["p_over"] / (dec["p_over"] + dec["p_under"])
    mv = main.apply(_move, axis=1).dropna() if len(main) else pd.Series(dtype=float)
    fl = []
    for side in ("over", "under"):
        f = x[x[f"flag_{side}"] & x["result"].isin(["win", "loss", "push"])]
        won = (f["result"] == ("win" if side == "over" else "loss"))
        fl += list(np.where(f["result"] == "push", 0.0, np.where(won, 1.0, -1.0)))
    return {
        "props": int(len(x)), "main_lines": int(len(main)), "ladder_rungs": int(len(x) - len(main)),
        "players": int(x["player_id"].nunique()), "games": int(x["game_id"].nunique()),
        "graded": int(len(dec)),
        "brier_ours": _num(D.brier(ours, hit), 5) if len(dec) else None,
        "brier_books": _num(D.brier(dec["fair_over"], hit), 5) if len(dec) else None,
        "moved_toward": _num((mv > 0).mean(), 3) if len(mv) else None,
        "moved_away": _num((mv < 0).mean(), 3) if len(mv) else None,
        "moved_points": _num(mv.mean(), 2) if len(mv) else None, "moved_n": int(len(mv)),
        "flagged": int((x["flag_over"] | x["flag_under"]).sum()),
        "book_gaps": int(x["book_gap"].map(lambda v: isinstance(v, str) and bool(v)).sum()) if "book_gap" in x else 0,
        "middles": int(x["middle"].fillna(0).astype(bool).sum()) if "middle" in x else 0,
        "ladder_gaps": int((x["ladder_gap"].abs() > LADDER_GAP).sum()) if "ladder_gap" in x else 0,
        "flag_record": f"{fl.count(1.0)}-{fl.count(-1.0)}-{fl.count(0.0)}",
        "new": _new_vs(dec, hit, ours),
        "calibration": D.calibration_table(ours, hit, np.r_[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]) if len(dec) else [],
        "calibration_books": D.calibration_table(dec["fair_over"], hit, np.r_[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]) if len(dec) else [],
    }


def _new_vs(dec, hit, ours) -> dict | None:
    """The new projections against the current ones and the books, on the props both priced."""
    if "pn_over" not in dec:
        return None
    m = dec["pn_over"].notna().to_numpy()
    if not m.any():
        return None
    new = dec["pn_over"][m] / (dec["pn_over"][m] + dec["pn_under"][m])
    return {"n": int(m.sum()), "brier_new": _num(D.brier(new, hit[m]), 5), "brier_ours": _num(D.brier(ours[m], hit[m]), 5),
            "brier_books": _num(D.brier(dec["fair_over"][m], hit[m]), 5)}


def _rec(r) -> dict:
    return {
        "g": r["game_id"], "id": r["player_id"], "n": r["player"], "t": r["team"], "pos": r["pos"], "s": r["stat"],
        "alt": bool(str(r["market"]).endswith("_alternate")), "L": _num(r["line"], 1), "mu": _num(r["proj"], 1),
        "po": _num(r["p_over"]), "pu": _num(r["p_under"]), "pp": _num(r["p_push"]), "fo": _num(r["fair_over"]),
        "pno": _num(r.get("pn_over")), "pnu": _num(r.get("pn_under")), "mun": _num(r.get("proj_new"), 1),
        "eo": _num(r["edge_over"]), "eu": _num(r["edge_under"]), "vo": _num(r["ev_over"]), "vu": _num(r["ev_under"]),
        "bk": _books(r["books"]), "fl": "over" if r["flag_over"] else "under" if r["flag_under"] else None,
        "wn": r["why_not"] if isinstance(r["why_not"], str) else "",
        "snap": r["snapshot"], "at": r["time"], "ko": r["commence"],
        "cL": _num(r.get("close_line"), 1), "cfo": _num(r.get("close_fair_over")),
        # market-only tags: bg = the two books' lines a step apart, mid = a middle, lg = ladder-rung
        # gap against the book's own main line (over terms), tb = that book, take = the side to take
        "bg": r.get("book_gap") if isinstance(r.get("book_gap"), str) and r.get("book_gap") else None,
        "mid": bool(r.get("middle")) if r.get("middle") is not None and not pd.isna(r.get("middle")) else False,
        "lg": _num(r.get("ladder_gap")), "tb": r.get("tag_book") if isinstance(r.get("tag_book"), str) and r.get("tag_book") else None,
        "take": r.get("take") if isinstance(r.get("take"), str) and r.get("take") else None,
        "y": _num(r.get("actual"), 1), "res": r.get("result") if isinstance(r.get("result"), str) else None,
    }


def build(out_dir=OUT) -> dict:
    log, res = PLOG.read_log(), PLOG.read_results()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = records(log, res)
    weeks = []
    if len(rec):
        for (season, week), x in rec.groupby(["season", "week"]):
            f = f"{int(season)}_w{int(week):02d}.json"
            body = {"season": int(season), "week": int(week), "summary": summary(x),
                    "props": [_rec(r) for r in x.sort_values(["game_id", "player", "stat", "market", "line"]).to_dict("records")]}
            _dump(out_dir / f, body)
            weeks.append({"file": f, "season": int(season), "week": int(week), **{k: v for k, v in body["summary"].items() if not k.startswith("calibration")}})
    ix = {"updated": PLOG._now(), "flags_on": _flags_on(), "weeks": weeks,
          "season": summary(rec) if len(rec) else {"props": 0}}
    _dump(out_dir / "index.json", ix)
    return ix


def _flags_on() -> bool:
    from .propmarket import FLAGS_ON
    return bool(FLAGS_ON)

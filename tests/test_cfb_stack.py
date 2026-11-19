"""The college market-residual margin (cfbmodel/config.STACK_MARGIN, Nov 19, 2026)."""
import numpy as np
import pandas as pd

from cfbmodel import config as C, model


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(C.FIRST_MODEL_SEASON, C.FIRST_TEST_SEASON + 3):
        for i in range(300):
            x = {f: rng.normal() for f in model.MARGIN_FEATS + model.TOTAL_FEATS}
            true = 3 * x["d_qb"] + 2 * x["d_pts"]
            line = 2 * x["d_pts"] + rng.normal(0, 1)
            rows.append({**x, "game_id": f"{s}_{i}", "season": s, "week": 1 + i % 12, "played": True,
                         "result": true + rng.normal(0, 10), "spread_line": line if i % 10 else np.nan,
                         "total": 55 + rng.normal(0, 12), "total_line": 55.0})
    return pd.DataFrame(rows)


def test_stacked_projection_is_line_plus_fit(monkeypatch):
    monkeypatch.setattr(C, "STACK_MARGIN", True)
    df = _frame()
    out, info = model.walk_forward(df)
    s = max(info)
    assert info[s]["stacked"] and "spread_line" in info[s]["stack_coefs"]
    o = out[out.season == s]
    train = df[(df.season < s) & df.spread_line.notna()]
    fit = model.market_residual(train, list(model.MARGIN_FEATS), s)
    hl = o.spread_line.notna()
    want = o.loc[hl, "spread_line"] + fit.predict(o.loc[hl, model.MARGIN_FEATS + ["spread_line"]].to_numpy(float))
    assert np.allclose(o.loc[hl, "proj_margin"], want)
    # no line: the raw model, as before
    assert np.allclose(o.loc[~hl, "proj_margin"], o.loc[~hl, "model_margin"])
    # the inputs the line lacks (the quarterback) earn weight; the one it carries (points) next to none
    c = info[s]["stack_coefs"]
    assert abs(c["d_qb"]) > 5 * abs(c["d_pts"])


def test_switch_off_restores_blend(monkeypatch):
    monkeypatch.setattr(C, "STACK_MARGIN", False)
    df = _frame(1)
    out, info = model.walk_forward(df)
    s = max(info)
    o = out[(out.season == s) & out.spread_line.notna()]
    assert not info[s]["stacked"]
    assert np.allclose(o.proj_margin, o.spread_line + o.w_margin * (o.model_margin - o.spread_line))

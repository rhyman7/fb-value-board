"""Books check for experiments/props_context_test.py: the context flags that passed on the projections,
added to the books' vig-free chance on 2025 DraftKings/FanDuel main lines (logit offset, c fitted on 2023-24).

    python experiments/props_context_books.py
"""
import sys, json
sys.path.insert(0, 'experiments'); sys.path.insert(0, '.')
import numpy as np, pandas as pd
import props_blend as PB, props_context_test as T
from nflmodel import config as C, data
ctx = T.context(data.load_games())
priced = pd.read_pickle(C.RAW / "props_blend_priced.pkl")
d = json.load(open('experiments/out/props_context_test.json'))
cands = sorted({(r["group"], r["stat"], r["flag"]) for r in d["passing_on_board"]})
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
out = []
for grp, stat, flag in cands:
    def fr(seasons):
        fl = pd.concat([priced[s][0] for s in seasons], ignore_index=True)
        fl = fl[(fl.stat == stat) & (fl.group == grp)].merge(ctx[["game_id", "team", flag]], on=["game_id", "team"], how="left")
        x = PB.frame(fl); x[flag] = x[flag].fillna(0); return x
    tr, te = fr((2023, 2024)), fr((2025,))
    if len(tr) < 200 or len(te) < 100 or tr[flag].sum() < 20:
        out.append((grp, stat, flag, len(tr), len(te), None, None, None)); continue
    z0, s, y = logit(tr.mkt.to_numpy()), tr[flag].to_numpy(float), tr.hit.to_numpy()
    c = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-(z0 + c * s))); g, h = (s * (y - p)).sum(), (s * s * p * (1 - p)).sum()
        if h <= 0: break
        c += g / h
    p = 1 / (1 + np.exp(-(logit(te.mkt.to_numpy()) + c * te[flag].to_numpy(float))))
    dd = pd.Series((p - te.hit) ** 2 - (te.mkt - te.hit) ** 2)
    out.append((grp, stat, flag, len(tr), len(te), round(c, 3), round(float(dd.mean()), 6), round(PB._se_by_game(dd.reset_index(drop=True), te.game_id.reset_index(drop=True)), 6)))
t = pd.DataFrame(out, columns=["group", "stat", "flag", "lines_2023_24", "lines_2025", "c", "brier_vs_book_2025", "se"])
print(t.to_string(index=False)); t.to_json('experiments/out/props_context_books.json', orient='records', indent=1)

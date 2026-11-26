"""Added description for experiments/market_shrug_test.py (Nov 26, 2026), written AFTER that test ran.

    PFF_DATA_DIR=<pff-data clone> python experiments/market_shrug_record.py

Not part of the pre-set rule and judges nothing. On the tuning seasons only (2014-18; the holdout is
not read), it restates Part 1 as a record: how the side the injury news favors did against the
closing and opening spreads, by what the line did between the opener and the close.
"""
import sys, warnings; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import new_absences_test as N, market_shrug_test as M
from nflmodel import model
base, rows, R, avail, op, pff = N.frames()
t,_ = N.team_counts(rows,R,avail)
f = N.attach(base,t).merge(op[["game_id","open_spread"]],on="game_id",how="left")
df = M.raw_model(f, list(model.MARGIN_FEATS))
df["news"]=df.per_starter*df.d_n_news; df["move"]=(df.spread_line-df.open_spread)*np.sign(df.news)
ng=(df.news.abs()>=1)&df.open_spread.notna()&df.spread_line.notna()&df.played
tn=df[ng&df.season.between(2014,2018)]
for lab,g in (("all",tn),("shrug",tn[tn.move<=0]),("moved against",tn[tn.move<0]),("no move",tn[tn.move==0]),("followed",tn[tn.move>=0.5])):
    c=np.sign(g.news)*(g.result-g.spread_line); o=np.sign(g.news)*(g.result-g.open_spread)
    print(lab, len(g), "news side vs close W-L-P", int((c>0).sum()),int((c<0).sum()),int((c==0).sum()), "| vs opener", int((o>0).sum()),int((o<0).sum()),int((o==0).sum()), "| by season shrug margin:", )
s=tn[tn.move<=0]; print(s.assign(c=np.sign(s.news)*(s.result-s.spread_line)).groupby("season").c.agg(["size","mean"]).round(2).to_string())

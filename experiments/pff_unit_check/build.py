"""Unit-rating drop vs production (descriptive). Rule stated to Ryan before the run, Nov 11 2026."""
import sys, glob, numpy as np, pandas as pd
PFF, D = sys.argv[1], sys.argv[2]
PO = {28:19,29:20,30:21,32:22}
UNITS = {"OL":({"T","G","C"},"off","grades_offense","snap_counts_total"),
         "REC":({"WR","TE"},"off","grades_pass_route","snap_counts_pass_route"),
         "RB":({"HB","FB"},"off","grades_offense","snap_counts_total"),
         "DL":({"DI","ED"},"def","grades_defense","snap_counts_defense"),
         "LB":({"LB"},"def","grades_defense","snap_counts_defense"),
         "DB":({"CB","S"},"def","grades_coverage_defense","snap_counts_coverage")}
NREG = {"OL":5,"REC":4,"RB":1,"DL":4,"LB":2,"DB":4}
def load(t):
    fs=[]
    for f in sorted(glob.glob(f"{PFF}/nfl/*/w*_{t}.csv.gz")):
        try: fs.append(pd.read_csv(f, low_memory=False))
        except Exception: pass
    return pd.concat(fs, ignore_index=True)
off, de = load("offense_summary"), load("defense_summary")
rows=[]
for u,(pos,side,g,s) in UNITS.items():
    src = off if side=="off" else de
    d = src[src.position.isin(pos)][["season","week","player_id","franchise_id",g,s]].copy()
    d.columns=["season","week","pid","fid","grade","snaps"]; d["unit"]=u; rows.append(d)
p = pd.concat(rows, ignore_index=True)
p["grade"]=pd.to_numeric(p.grade,errors="coerce"); p["snaps"]=pd.to_numeric(p.snaps,errors="coerce")
p=p.dropna(subset=["snaps","grade"]); p=p[p.snaps>0]
p["week"]=p.week.astype(int).map(lambda w: PO.get(w,w)); p=p[p.week.between(1,22)]
p["t"]=p.season.astype(int)*40+p.week  # 40 "weeks" a year: offseason gap = 18+
p=p.sort_values(["unit","pid","t"]).reset_index(drop=True)
# prior rating: decayed snap-weighted mean of earlier games, shrunk to 55 with 200 snaps
HL, K, BASE = 20.0, 200.0, 55.0
pri=np.empty(len(p))
for _,ix in p.groupby(["unit","pid"]).indices.items():
    t=p.t.values[ix]; g=p.grade.values[ix]; s=p.snaps.values[ix]
    for j in range(len(ix)):
        if j==0: pri[ix[j]]=BASE; continue
        w=s[:j]*0.5**((t[j]-t[:j])/HL)
        pri[ix[j]]=(np.sum(w*g[:j])+K*BASE)/(np.sum(w)+K)
p["prior"]=pri
p["hist"]=p.groupby(["unit","pid"]).cumcount()
# franchise -> nflverse team via PFF games matched on scores
pg=load("games"); pg["week"]=pg.week.astype(int).map(lambda w: PO.get(w,w))
g=pd.read_csv(f"{D}/games.csv"); g=g[(g.season>=2014)&g.home_score.notna()]
m=pg.merge(g,left_on=["season","week","score_home_team","score_away_team"],right_on=["season","week","home_score","away_score"])
fm=pd.concat([m[["season","home_franchise_id","home_team"]].set_axis(["season","fid","team"],axis=1),
              m[["season","away_franchise_id","away_team"]].set_axis(["season","fid","team"],axis=1)])
fm=fm.groupby(["season","fid"]).team.agg(lambda x:x.value_counts().index[0]).reset_index()
p=p.merge(fm,on=["season","fid"],how="left"); print("unmapped",p.team.isna().mean())
p=p.dropna(subset=["team"])
p.to_parquet(f"{D}/player_games.parquet")
# unit lineup rating per team-game
p["wp"]=p.snaps*p.prior
ug=p.groupby(["season","week","team","unit"]).agg(wp=("wp","sum"),snaps=("snaps","sum")).reset_index()
ug["rating"]=ug.wp/ug.snaps
ug=ug.sort_values(["team","unit","season","week"])
def trail(x): return x.shift(1).rolling(8,min_periods=3).mean()
ug["usual"]=ug.groupby(["team","unit","season"]).rating.transform(trail)
ug["drop"]=ug.rating-ug.usual
# regulars missing: top-N by snaps over prior games this season who took no snaps
ps=p[["season","week","team","unit","pid","snaps"]]
miss=[]
for (s,tm,u),d in ps.groupby(["season","team","unit"]):
    wk=sorted(d.week.unique()); piv=d.pivot_table(index="pid",columns="week",values="snaps",aggfunc="sum").fillna(0)
    for i,w in enumerate(wk):
        if i<3: continue
        prev=piv[wk[max(0,i-8):i]]; rec=prev.iloc[:,-3:].sum(axis=1)
        reg=rec.sort_values(ascending=False).head(NREG[u]); reg=reg[reg>0].index
        miss.append((s,w,tm,u,int((piv.loc[reg,w]==0).sum())))
miss=pd.DataFrame(miss,columns=["season","week","team","unit","n_missing"])
ug=ug.merge(miss,on=["season","week","team","unit"],how="left")
ug.to_parquet(f"{D}/unit_games.parquet"); print(ug.groupby("unit")[["rating","drop","n_missing"]].describe().T.to_string())

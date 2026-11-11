import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
D=sys.argv[1]
ug=pd.read_parquet(f"{D}/unit_games.parquet")
g=pd.read_csv(f"{D}/games.csv"); g=g[(g.season>=2015)&g.home_score.notna()]
rows=[]
for s in range(2015,2027):
    b=pd.read_parquet(f"{D}/pbp_{s}.parquet",columns=["game_id","season","week","posteam","defteam","pass","rush","epa","sack","yards_gained","qb_dropback","passer_player_id"])
    b=b[((b["pass"]==1)|(b["rush"]==1))&b.epa.notna()&b.posteam.notna()]
    a=b.groupby(["game_id","season","week","posteam","defteam"]).apply(lambda d: pd.Series({
        "epa":d.epa.mean(),"ypp":d.yards_gained.mean(),
        "pass_epa":d.epa[d["pass"]==1].mean(),"rush_epa":d.epa[d["rush"]==1].mean(),
        "sack_rate":100*d.sack[d["pass"]==1].mean(),
        "qb":d.passer_player_id.dropna().mode().iloc[0] if d.passer_player_id.notna().any() else None})).reset_index()
    rows.append(a)
o=pd.concat(rows,ignore_index=True).rename(columns={"posteam":"team","defteam":"opp"})
gl=pd.concat([g.assign(team=g.home_team,pts=g.home_score,imp=g.total_line/2+g.spread_line/2,ats=(g.home_score-g.away_score)-g.spread_line)[["game_id","team","pts","imp","ats","game_type"]],
              g.assign(team=g.away_team,pts=g.away_score,imp=g.total_line/2-g.spread_line/2,ats=-((g.home_score-g.away_score)-g.spread_line))[["game_id","team","pts","imp","ats","game_type"]]])
o=o.merge(gl,on=["game_id","team"])
o["week"]=o.week.astype(int); o=o.sort_values(["season","week"])
M=["pts","epa","ypp","pass_epa","rush_epa","sack_rate"]
tr=lambda x:x.shift(1).rolling(8,min_periods=3).mean()
for m in M:
    o[m+"_own"]=o.groupby(["season","team"])[m].transform(tr)
    o[m+"_opp"]=o.groupby(["season","opp"])[m].transform(tr)
    o[m+"_lg"]=o.groupby("season")[m].transform("mean")
    o[m+"_r"]=o[m]-(o[m+"_own"]+o[m+"_opp"]-o[m+"_lg"])
o["qb_usual"]=o.groupby(["season","team"]).qb.transform(lambda x: x.shift(1).expanding().apply(lambda v: 0, raw=False) if False else x.shift(1))
# usual QB = most common passer over prior games this season
def usualqb(x):
    out=[];
    for i in range(len(x)):
        out.append(x.iloc[:i].mode().iloc[0] if i>0 and x.iloc[:i].notna().any() else None)
    return pd.Series(out,index=x.index)
o["qb_usual"]=o.groupby(["season","team"]).qb.transform(usualqb)
o["qb_chg"]=((o.qb!=o.qb_usual)&o.qb_usual.notna()).astype(int)
o["mkt_r"]=o.pts-o.imp
w=ug.pivot_table(index=["season","week","team"],columns="unit",values="drop").reset_index()
n=ug.pivot_table(index=["season","week","team"],columns="unit",values="n_missing").add_prefix("n_").reset_index()
w=w.merge(n,on=["season","week","team"])
f=o.merge(w,on=["season","week","team"]).merge(w.rename(columns={c:"opp_"+c for c in w.columns if c not in("season","week")}).rename(columns={"opp_team":"opp"}),on=["season","week","opp"])
f=f.dropna(subset=["pts_r","OL","opp_DL"])
last=f.groupby("season").week.transform(lambda x: 18 if x.max()>=18 and x.name>=2021 else 17)
f["final_wk"]=((f.game_type=="REG")&(f.week==np.where(f.season>=2021,18,17))).astype(int)
f["ts"]=f.season.astype(str)+f.team
f.to_parquet(f"{D}/frame.parquet")
OFFU=["OL","REC","RB"]; DEFU=["opp_DL","opp_LB","opp_DB"]
def run(d,label):
    print(f"\n===== {label}: {len(d)} team-games =====")
    print("-- joint regression: offence outcome vs own-offence unit drops and opposing-defence unit drops (per 1 grade point), QB change controlled")
    res=[]
    for m in ["pts_r","epa_r","pass_epa_r","rush_epa_r","sack_rate_r","mkt_r","ats"]:
        r=smf.ols(f"{m} ~ OL+REC+RB+opp_DL+opp_LB+opp_DB+qb_chg",d).fit(cov_type="cluster",cov_kwds={"groups":d.game_id.astype("category").cat.codes})
        res.append(pd.Series({k:f"{r.params[k]:+.3f} ({r.tvalues[k]:+.1f})" for k in OFFU+DEFU+["qb_chg"]},name=m))
    print(pd.DataFrame(res).to_string())
    print("-- SD of drop:",d[OFFU+DEFU].std().round(2).to_dict())
    print("-- regulars missing -> mean points vs expected (pts_r), vs market team total (mkt_r), n")
    for u,sign in [("OL",1),("REC",1),("DL",-1),("LB",-1),("DB",-1)]:
        col="n_"+u if sign==1 else "opp_n_"+u
        t=d.assign(k=d[col].clip(upper=2)).groupby("k").agg(n=("pts_r","size"),pts_r=("pts_r","mean"),se=("pts_r","sem"),epa_r=("epa_r","mean"),mkt_r=("mkt_r","mean"),mkt_se=("mkt_r","sem"),ats=("ats","mean"),cover=("ats",lambda x:(x>0).sum()/((x!=0).sum())))
        print(("own " if sign==1 else "OPPOSING ")+u); print(t.round(3).to_string())
    print("-- worst 10% drop games per unit: pts_r, mkt_r (offence scoring for own units; offence scoring vs that defence for opp units)")
    for u in OFFU+DEFU:
        q=d[u].quantile(0.10); b=d[d[u]<=q]
        print(f"{u:7s} cut {q:+.2f} n={len(b)} drop avg {b[u].mean():+.2f} | pts_r {b.pts_r.mean():+.2f}±{b.pts_r.sem():.2f} | epa_r {b.epa_r.mean():+.3f}±{b.epa_r.sem():.3f} | vs mkt total {b.mkt_r.mean():+.2f}±{b.mkt_r.sem():.2f} | ATS {b.ats.mean():+.2f}±{b.ats.sem():.2f} cover {(b.ats>0).sum()}/{(b.ats!=0).sum()}")
h=f[f.season<=2025]
run(h[h.final_wk==0],"2015-2025, final regular-season week left out")
run(h,"2015-2025, all games")
run(f[f.season==2026],"2026 weeks 4-8")

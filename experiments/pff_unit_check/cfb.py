import sys, glob, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
PFF, D = sys.argv[1], sys.argv[2]
UNITS = {"OL":({"T","G","C"},"off","grades_offense","snap_counts_total"),
         "REC":({"WR","TE"},"off","grades_pass_route","snap_counts_pass_route"),
         "RB":({"HB","FB"},"off","grades_offense","snap_counts_total"),
         "DL":({"DI","ED"},"def","grades_defense","snap_counts_defense"),
         "LB":({"LB"},"def","grades_defense","snap_counts_defense"),
         "DB":({"CB","S"},"def","grades_coverage_defense","snap_counts_coverage")}
NREG={"OL":5,"REC":4,"RB":1,"DL":4,"LB":2,"DB":4}
def load(t,cols=None):
    fs=[]
    for f in sorted(glob.glob(f"{PFF}/ncaa/*/w*_{t}.csv.gz")):
        try: fs.append(pd.read_csv(f, low_memory=False, usecols=cols))
        except Exception: pass
    return pd.concat(fs, ignore_index=True)
g=pd.read_parquet(f"{D}/cfb_games.parquet"); g=g[g.home_score.notna()]
pg=load("games")
m=pg.merge(g,left_on=["season","score_home_team","score_away_team"],right_on=["season","home_score","away_score"],suffixes=("_p",""))
m=m[(m.week_p-m.week).abs()<=1]
fm=pd.concat([m[["season","home_franchise_id","home_team"]].set_axis(["season","fid","team"],axis=1),m[["season","away_franchise_id","away_team"]].set_axis(["season","fid","team"],axis=1)])
vc=fm.groupby(["season","fid","team"]).size().reset_index(name="n").sort_values("n")
fm=vc.drop_duplicates(["season","fid"],keep="last"); fm=fm[fm.n>=4][["season","fid","team"]]
# PFF game -> our game id / week, per team
gm=pd.concat([m[["season","week_p","home_franchise_id","game_id","week","postseason"]].set_axis(["season","pweek","fid","game_id","gweek","post"],axis=1),
              m[["season","week_p","away_franchise_id","game_id","week","postseason"]].set_axis(["season","pweek","fid","game_id","gweek","post"],axis=1)]).drop_duplicates(["season","pweek","fid"])
print("FBS team-seasons mapped",len(fm), "games matched",m.game_id.nunique(),"of",len(g))
off=load("offense_summary",["season","week","player_id","franchise_id","position","grades_offense","grades_pass_route","snap_counts_total","snap_counts_pass_route"])
de=load("defense_summary",["season","week","player_id","franchise_id","position","grades_defense","grades_coverage_defense","snap_counts_defense","snap_counts_coverage"])
rows=[]
for u,(pos,side,gc,sc) in UNITS.items():
    src=off if side=="off" else de
    d=src[src.position.isin(pos)][["season","week","player_id","franchise_id",gc,sc]].copy(); d.columns=["season","week","pid","fid","grade","snaps"]; d["unit"]=u; rows.append(d)
p=pd.concat(rows,ignore_index=True)
p["grade"]=pd.to_numeric(p.grade,errors="coerce"); p["snaps"]=pd.to_numeric(p.snaps,errors="coerce"); p=p.dropna(subset=["grade","snaps"]); p=p[p.snaps>0]
p["t"]=p.season*40+p.week
keep=set(p.merge(fm,on=["season","fid"]).pid)   # anyone who ever played for an FBS team: full history incl. other divisions
p=p[p.pid.isin(keep)]
HL,K,BASE=20.0,200.0,55.0
hist={k:(d.t.values,d.grade.values,d.snaps.values) for k,d in p.sort_values("t").groupby(["unit","pid"])}
def rating(u,pid,t):
    tt,gg,s=hist[(u,pid)]; mm=tt<t
    if not mm.any(): return BASE
    w=s[mm]*0.5**((t-tt[mm])/HL); return (np.sum(w*gg[mm])+K*BASE)/(np.sum(w)+K)
p=p.merge(fm,on=["season","fid"])
out=[]
for (s,tm,fid,u),d in p.groupby(["season","team","fid","unit"]):
    wk=sorted(d.week.unique()); piv=d.pivot_table(index="pid",columns="week",values="snaps",aggfunc="sum").fillna(0); sh=piv/piv.sum(axis=0)
    for i,w in enumerate(wk):
        if i<3: continue
        t=s*40+w; prev=wk[max(0,i-8):i]; wt=0.5**(np.arange(len(prev))[::-1]/4.0)
        usual=(sh[prev]*wt).sum(axis=1)/wt.sum(); usual=usual[usual>0]; act=sh[w]; act=act[act>0]
        r={q:rating(u,q,t) for q in set(usual.index)|set(act.index)}
        ur=sum(usual[q]*r[q] for q in usual.index)/usual.sum(); ar=sum(act[q]*r[q] for q in act.index)
        reg=usual.sort_values(ascending=False).head(NREG[u]).index
        out.append((s,w,fid,tm,u,ar-ur,int((piv.loc[reg,w]==0).sum())))
ug=pd.DataFrame(out,columns=["season","pweek","fid","team","unit","drop","n_missing"]).merge(gm,on=["season","pweek","fid"])
ug.to_parquet(f"{D}/cfb_unit_games.parquet")
w=ug.pivot_table(index=["game_id","team"],columns="unit",values="drop").reset_index().merge(ug.pivot_table(index=["game_id","team"],columns="unit",values="n_missing").add_prefix("n_").reset_index(),on=["game_id","team"])
sgn=np.sign(np.corrcoef(g.dropna(subset=["spread_line"]).spread_line,(g.home_score-g.away_score)[g.spread_line.notna()])[0,1]); print("spread sign (home margin corr)",sgn)
g["sl"]=g.spread_line*sgn  # home expected margin
o=pd.concat([g.assign(team=g.home_team,opp=g.away_team,pts=g.home_score,imp=g.total_line/2+g.sl/2,ats=(g.home_score-g.away_score)-g.sl),
             g.assign(team=g.away_team,opp=g.home_team,pts=g.away_score,imp=g.total_line/2-g.sl/2,ats=-((g.home_score-g.away_score)-g.sl))])[["game_id","season","week","postseason","kickoff","team","opp","pts","imp","ats"]]
o=o.sort_values("kickoff"); tr=lambda x:x.shift(1).rolling(8,min_periods=3).mean()
o["own"]=o.groupby(["season","team"]).pts.transform(tr); o["oppa"]=o.groupby(["season","opp"]).pts.transform(tr); o["lg"]=o.groupby("season").pts.transform("mean")
o["pts_r"]=o.pts-(o.own+o.oppa-o.lg); o["mkt_r"]=o.pts-o.imp
f=o.merge(w,on=["game_id","team"]).merge(w.rename(columns={c:"opp_"+c for c in w.columns if c!="game_id"}).rename(columns={"opp_team":"opp"}),on=["game_id","opp"])
f=f.dropna(subset=["OL","opp_DL"]); f.to_parquet(f"{D}/cfb_frame.parquet")
OFFU=["OL","REC","RB"]; DEFU=["opp_DL","opp_LB","opp_DB"]
def run(d,label):
    print(f"\n===== {label}: {len(d)} team-games =====")
    if len(d) < 50: return
    for mm in ["pts_r","mkt_r","ats"]:
        dd=d.dropna(subset=[mm]+OFFU+DEFU); r=smf.ols(f"{mm} ~ OL+REC+RB+opp_DL+opp_LB+opp_DB",dd).fit(cov_type="cluster",cov_kwds={"groups":dd.game_id.astype("category").cat.codes})
        print(f"{mm:6s} n={len(dd)} "+"  ".join(f"{k} {r.params[k]:+.2f} ({r.tvalues[k]:+.1f})" for k in OFFU+DEFU))
    print("SD of drop:",d[OFFU+DEFU].std().round(2).to_dict())
    for u,own in [("OL",1),("REC",1),("DL",0),("LB",0),("DB",0)]:
        col="n_"+u if own else "opp_n_"+u
        t=d.assign(k=d[col].clip(upper=2)).groupby("k").agg(n=("pts","size"),pts_r=("pts_r","mean"),se=("pts_r","sem"),mkt_r=("mkt_r","mean"),mkt_se=("mkt_r","sem"),cover=("ats",lambda x:(x>0).sum()/max((x.notna()&(x!=0)).sum(),1)),nats=("ats",lambda x:(x.notna()&(x!=0)).sum()))
        print(("own " if own else "OPPOSING ")+u+" regulars missing"); print(t.round(3).to_string())
reg=f[f.postseason==0]
run(reg[reg.season<=2025],"College regular season 2017-2025")
run(reg[reg.season==2026],"College 2026")

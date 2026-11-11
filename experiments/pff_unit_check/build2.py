import sys, numpy as np, pandas as pd
D=sys.argv[1]
p=pd.read_parquet(f"{D}/player_games.parquet")
HL,K,BASE=20.0,200.0,55.0
NREG={"OL":5,"REC":4,"RB":1,"DL":4,"LB":2,"DB":4}
hist={k:(d.t.values,d.grade.values,d.snaps.values) for k,d in p.sort_values("t").groupby(["unit","pid"])}
def rating(u,pid,t):
    tt,g,s=hist[(u,pid)]; m=tt<t
    if not m.any(): return BASE
    w=s[m]*0.5**((t-tt[m])/HL); return (np.sum(w*g[m])+K*BASE)/(np.sum(w)+K)
out=[]
for (s,tm,u),d in p.groupby(["season","team","unit"]):
    wk=sorted(d.week.unique()); piv=d.pivot_table(index="pid",columns="week",values="snaps",aggfunc="sum").fillna(0)
    sh=piv/piv.sum(axis=0)
    for i,w in enumerate(wk):
        if i<3: continue
        t=s*40+w; prev=wk[max(0,i-8):i]
        wt=0.5**(np.arange(len(prev))[::-1]/4.0)
        usual=(sh[prev]*wt).sum(axis=1)/wt.sum(); usual=usual[usual>0]
        act=sh[w]; act=act[act>0]
        r={pid:rating(u,pid,t) for pid in set(usual.index)|set(act.index)}
        ur=sum(usual[i_]*r[i_] for i_ in usual.index)/usual.sum()
        ar=sum(act[i_]*r[i_] for i_ in act.index)
        reg=usual.sort_values(ascending=False).head(NREG[u]).index
        out.append((s,w,tm,u,ar,ur,ar-ur,int((piv.loc[reg,w]==0).sum())))
ug=pd.DataFrame(out,columns=["season","week","team","unit","rating","usual","drop","n_missing"])
ug.to_parquet(f"{D}/unit_games.parquet")
print(ug.groupby("unit")[["rating","drop","n_missing"]].agg(["mean","std","min","max"]).round(2).to_string())
print(ug.groupby(["unit","n_missing"])["drop"].agg(["size","mean"]).round(2).to_string())

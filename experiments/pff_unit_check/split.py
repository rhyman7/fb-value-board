import sys,pandas as pd,numpy as np
f=pd.read_parquet(sys.argv[1]+"/frame.parquet")
f["off_n"]=f.n_OL+f.n_REC; f["def_n"]=f.opp_n_DL+f.opp_n_LB+f.opp_n_DB
def line(b):
    c=(b.ats>0).sum(); n=(b.ats!=0).sum()
    return f"n={len(b):4d} pts_r {b.pts_r.mean():+.2f}±{b.pts_r.sem():.2f} | vs mkt total {b.mkt_r.mean():+.2f}±{b.mkt_r.sem():.2f} | cover {c}/{n} {c/n:.3f}"
per={"2015-18":(2015,2018),"2019-25":(2019,2025),"2026":(2026,2026)}
for lab,cond in [("own OL 2+ out",lambda d:d.n_OL>=2),("own OL 3+ out",lambda d:d.n_OL>=3),("opp LB 1+ out",lambda d:d.opp_n_LB>=1),("opp DB 1 out",lambda d:d.opp_n_DB==1),
                 ("offence (OL+REC) 3+ out",lambda d:d.off_n>=3),("offence 0 out",lambda d:d.off_n==0),("opp defence 3+ out",lambda d:d.def_n>=3),("opp defence 0 out",lambda d:d.def_n==0)]:
    print(lab)
    for p,(a,b) in per.items():
        d=f[f.season.between(a,b)&((f.final_wk==0))]; print("  ",p,line(d[cond(d)]))

import sys,pandas as pd,numpy as np
f=pd.read_parquet(sys.argv[1]+"/cfb_frame.parquet"); f=f[f.postseason==0]
def line(b):
    a=b.ats.dropna(); c=(a>0).sum(); n=(a!=0).sum()
    return f"n={len(b):5d} pts_r {b.pts_r.mean():+.2f}±{b.pts_r.sem():.2f} | vs mkt total {b.mkt_r.mean():+.2f}±{b.mkt_r.sem():.2f} | cover {c}/{n} {c/max(n,1):.3f}"
f["def_n"]=f.opp_n_DL+f.opp_n_LB+f.opp_n_DB; f["off_n"]=f.n_OL+f.n_REC
for lab,cond in [("opp DB 1+ out",lambda d:d.opp_n_DB>=1),("opp DL 2+ out",lambda d:d.opp_n_DL>=2),("opp defence 3+ out",lambda d:d.def_n>=3),("opp defence 0 out",lambda d:d.def_n==0),
                 ("own REC 1+ out",lambda d:d.n_REC>=1),("own OL 2+ out",lambda d:d.n_OL>=2),("own offence 3+ out",lambda d:d.off_n>=3)]:
    print(lab)
    for p,(a,b) in {"2017-21":(2017,2021),"2022-25":(2022,2025),"2026":(2026,2026)}.items():
        d=f[f.season.between(a,b)]; print("  ",p,line(d[cond(d)]))
print(f.groupby("season").size().to_dict())

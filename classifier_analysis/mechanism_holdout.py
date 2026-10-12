"""Hold out whole mechanisms (8 groups of families) instead of single families.

    python3 classifier_analysis/mechanism_holdout.py      # from spar-analysis/
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from classifier_analysis import common as C
from classifier_analysis.full_extra import fit_designs, with_rule, score, designs
D,gc=C.load(); D=with_rule(D)
G={'dp_allreduce':['train_ddp','train_fsdp','train_lora','train_contrastive','train_diffusion','train_accum','train_accumsync','trainmem_fsdp'],
 'model_parallel':['train_tp','train_pipe','train_moe'],
 'low_comm':['train_localsgd','train_streamdiloco','train_decoupdiloco','train_swarm'],
 'rl':['train_impala','train_apexrl','train_ppodist'],
 'llm_serving':[f for f in D.family.unique() if f.startswith(('inferserve','infermem','infershard','inferhf','inferrep'))],
 'replicated_inf':['inferasr_whisper','infercnn_rn','inferdiff_sdxl','inferembed_bge','inferembed_clip','infervis_vitl'],
 'balanced_hpc':['nonml_fftshard','nonml_stencil','nonml_nccl','nonml_comm','inferrec_dlrm'],
 'other_compute':['nonml_hpc','nonml_hpl','nonml_himem','nonml_lowpower','nonml_sat']}
m={f:g for g,fs in G.items() for f in fs}
D['mech']=D.family.map(m); assert D.mech.notna().all(), D[D.mech.isna()].family.unique()
parts=[]
for g in G:
    te=D.mech.values==g
    parts.append(fit_designs(D[~te],D[te],gc).assign(mech=g)); print(g,flush=True)
P=pd.concat(parts); P.to_parquet('classifier_analysis/results/full/mechanism_holdout_preds.parquet',index=False)
S=score(P); S.to_csv('classifier_analysis/results/full/mechanism_holdout.csv',index=False)
pd.set_option('display.width',250); print(S.to_string(index=False))
per=[P.assign(c=c.values).groupby(['is_train','mech']).c.mean().rename(k) for k,c in designs(P).items()]
F=pd.concat(per,axis=1).round(2); F.to_csv('classifier_analysis/results/full/mechanism_holdout_by_group.csv'); print(F.to_string())

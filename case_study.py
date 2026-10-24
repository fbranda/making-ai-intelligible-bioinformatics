"""Synthetic regulatory-genomics case study for 'Making AI Intelligible in Bioinformatics'.
Runs all six practices on a simulated TF-binding task with planted ground truth.
Usage: python case_study.py  (writes results.json, model.joblib, and curve_*.npy in the cwd)
"""
import json, itertools, time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
import shap
from lime.lime_tabular import LimeTabularExplainer

rng = np.random.default_rng(2026)
BASES = np.array(list("ACGT"))
MOTIF = "GATAAG"
L, K = 100, 4
N_CHR, LOCI_PER_CHR, PER_LOCUS = 10, 30, 40
KMERS = ["".join(p) for p in itertools.product("ACGT", repeat=K)]
KIDX = {k: i for i, k in enumerate(KMERS)}
MOTIF_K = [MOTIF[i:i + K] for i in range(len(MOTIF) - K + 1)]  # GATA, ATAA, TAAG
MOTIF_IDX = [KIDX[k] for k in MOTIF_K]

# ---------- 1. simulate data ----------
def make_data():
    seqs, y, chrom, locus = [], [], [], []
    lid = 0
    for c in range(1, N_CHR + 1):
        for _ in range(LOCI_PER_CHR):
            tmpl = rng.integers(0, 4, L)
            p_loc = rng.beta(0.6, 0.6)               # locus-specific base rate (confounder)
            for _ in range(PER_LOCUS):
                s = tmpl.copy()
                mut = rng.random(L) < 0.10
                s[mut] = rng.integers(0, 4, mut.sum())
                label = int(rng.random() < p_loc)
                plant = rng.random() < (0.70 if label else 0.10)  # motif imperfect signal
                st = "".join(BASES[s])
                if plant:
                    pos = rng.integers(0, L - len(MOTIF) + 1)
                    st = st[:pos] + MOTIF + st[pos + len(MOTIF):]
                seqs.append(st); y.append(label); chrom.append(c); locus.append(lid)
            lid += 1
    return seqs, np.array(y), np.array(chrom), np.array(locus)

def kmer_counts(seqs):
    X = np.zeros((len(seqs), len(KMERS)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(len(s) - K + 1):
            X[i, KIDX[s[j:j + K]]] += 1
    return X

seqs, y, chrom, locus = make_data()
X = kmer_counts(seqs)
has_motif = np.array([MOTIF in s for s in seqs])
R = {"n": int(len(y)), "prevalence": float(y.mean())}

# splits
idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
row_tr, row_te = idx[:cut], idx[cut:]
tr = np.isin(chrom, [1, 2, 3, 4, 5, 6]); cal = np.isin(chrom, [7, 8]); te = np.isin(chrom, [9, 10])

def gbm(seed=0):
    return GradientBoostingClassifier(n_estimators=120, max_depth=3, subsample=0.5,
                                      learning_rate=0.1, random_state=seed)

# ---------- 2. Practice 2: benchmark interrogation ----------
t0 = time.time()
m_leak = gbm().fit(X[row_tr], y[row_tr])
auc_leak = roc_auc_score(y[row_te], m_leak.predict_proba(X[row_te])[:, 1])
m_hon = gbm().fit(X[tr], y[tr])
auc_hon = roc_auc_score(y[te], m_hon.predict_proba(X[te])[:, 1])
# label-shuffled controls (same pipeline)
ysh = rng.permutation(y[row_tr]); auc_sh_row = roc_auc_score(y[row_te], gbm().fit(X[row_tr], ysh).predict_proba(X[row_te])[:, 1])
ysh2 = rng.permutation(y[tr]); auc_sh_chr = roc_auc_score(y[te], gbm().fit(X[tr], ysh2).predict_proba(X[te])[:, 1])
# baseline 1: locus-identity prior (knows only which locus a row came from)
prior = {l: y[row_tr][locus[row_tr] == l].mean() for l in np.unique(locus[row_tr])}
auc_prior_row = roc_auc_score(y[row_te], [prior[l] for l in locus[row_te]])
auc_prior_chr = 0.5  # unseen loci -> constant prediction
# baseline 2: motif-presence logistic regression (3 k-mer counts)
lr_row = LogisticRegression().fit(X[row_tr][:, MOTIF_IDX], y[row_tr])
auc_lr_row = roc_auc_score(y[row_te], lr_row.predict_proba(X[row_te][:, MOTIF_IDX])[:, 1])
lr_chr = LogisticRegression().fit(X[tr][:, MOTIF_IDX], y[tr])
auc_lr_chr = roc_auc_score(y[te], lr_chr.predict_proba(X[te][:, MOTIF_IDX])[:, 1])
R["P2"] = dict(gbm_row=auc_leak, gbm_chrom=auc_hon, shuffled_row=auc_sh_row, shuffled_chrom=auc_sh_chr,
               locus_prior_row=auc_prior_row, locus_prior_chrom=auc_prior_chr,
               motif_lr_row=auc_lr_row, motif_lr_chrom=auc_lr_chr)
print("P2", R["P2"], time.time() - t0, flush=True)

# ---------- 3. Practice 3: attributions (MLP: IG, KernelSHAP, LIME, ablation; GBM: TreeSHAP) ----------
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
Z = (X - mu) / sd
def mlp(seed): return MLPClassifier((64, 32), alpha=1e-2, max_iter=200, early_stopping=True, random_state=seed)
mlps = [mlp(s).fit(Z[tr], y[tr]) for s in range(5)]
m0 = mlps[0]
auc_mlp = roc_auc_score(y[te], m0.predict_proba(Z[te])[:, 1])
auc_ens = roc_auc_score(y[te], np.mean([m.predict_proba(Z[te])[:, 1] for m in mlps], 0))
R["P3_models"] = dict(mlp_chrom_auc=auc_mlp, ensemble_chrom_auc=auc_ens)
print("MLP", auc_mlp, auc_ens, flush=True)

def mlp_logit_and_grad(m, Zb):
    a = Zb; acts = []
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        pre = a @ W + b; acts.append(pre > 0); a = np.maximum(pre, 0)
    out = a @ m.coefs_[-1] + m.intercepts_[-1]
    g = np.ones((len(Zb), 1)) @ m.coefs_[-1].T
    for W, mask in zip(m.coefs_[-2::-1], acts[::-1]):
        g = (g * mask) @ W.T
    return out[:, 0], g
def ig(m, z, base, steps=50):
    alphas = np.linspace(0, 1, steps)[:, None]
    path = base + alphas * (z - base)
    _, g = mlp_logit_and_grad(m, path)
    return (z - base) * g.mean(0)

# evaluation set: held-out true binders that carry the planted motif
cand = np.where(te & (y == 1) & has_motif)[0]
ev = rng.choice(cand, 80, replace=False)
base = np.zeros(Z.shape[1])  # training-mean baseline in standardised space
bgZ = Z[tr][rng.choice(tr.sum(), 50, replace=False)]
f_prob = lambda A: m0.predict_proba(A)[:, 1]
ks = shap.KernelExplainer(f_prob, shap.kmeans(bgZ, 10))
lime_ex = LimeTabularExplainer(Z[tr], mode="classification", discretize_continuous=False, random_state=0)
A = {k: [] for k in ["IG", "KernelSHAP", "LIME", "Ablation", "TreeSHAP (GBM)"]}
tree_ex = shap.TreeExplainer(m_hon)
tsh = tree_ex.shap_values(X[ev])
t1 = time.time()
for n, i in enumerate(ev):
    z = Z[i]
    A["IG"].append(ig(m0, z, base))
    A["KernelSHAP"].append(ks.shap_values(z[None, :], nsamples=300, silent=True).reshape(-1))
    e = lime_ex.explain_instance(z, lambda B: m0.predict_proba(B), num_features=256, num_samples=1500, labels=(1,))
    v = np.zeros(256)
    for f, w in e.as_map()[1]: v[f] = w
    A["LIME"].append(v)
    p0 = f_prob(z[None])[0]; zz = np.tile(z, (256, 1)); zz[np.arange(256), np.arange(256)] = 0.0
    A["Ablation"].append(p0 - f_prob(zz))
    A["TreeSHAP (GBM)"].append(tsh[n])
    if n % 20 == 0: print("attr", n, time.time() - t1, flush=True)
A = {k: np.array(v) for k, v in A.items()}

TOPK = 10
def recovery(M):
    out = []
    for r in M:
        top = np.argsort(-np.abs(r))[:TOPK]
        out.append(np.mean([m in top for m in MOTIF_IDX]))
    return float(np.mean(out))
rec = {k: recovery(v) for k, v in A.items()}
rec["Random ranking (expected)"] = TOPK / 256
rec["Random ranking (expected)"] = float(TOPK / 256)
names = list(A.keys())
agree = np.zeros((len(names), len(names))); jacc = np.zeros_like(agree)
for a, b in itertools.product(range(len(names)), repeat=2):
    sp, jj = [], []
    for r1, r2 in zip(A[names[a]], A[names[b]]):
        sp.append(spearmanr(np.abs(r1), np.abs(r2))[0])
        t1_, t2_ = set(np.argsort(-np.abs(r1))[:TOPK]), set(np.argsort(-np.abs(r2))[:TOPK])
        jj.append(len(t1_ & t2_) / len(t1_ | t2_))
    agree[a, b], jacc[a, b] = np.nanmean(sp), np.mean(jj)

# faithfulness: deletion test on the MLP (replace top-k features by training mean)
def deletion(M, kmax=20):
    curves = []
    for r, i in zip(M, ev):
        order = np.argsort(-r)
        z = Z[i].copy(); c = [f_prob(z[None])[0]]
        for k in range(kmax):
            z[order[k]] = 0.0; c.append(f_prob(z[None])[0])
        curves.append(c)
    return np.mean(curves, 0)
del_curves = {k: deletion(A[k]) for k in ["IG", "KernelSHAP", "LIME", "Ablation"]}
def deletion_gbm(M, kmax=20, random_order=False):
    curves = []
    for r, i in zip(M, ev):
        order = rng.permutation(256) if random_order else np.argsort(-r)
        x = X[i].copy(); c = [m_hon.predict_proba(x[None])[0, 1]]
        for k in range(kmax):
            x[order[k]] = mu[order[k]]; c.append(m_hon.predict_proba(x[None])[0, 1])
        curves.append(c)
    return np.mean(curves, 0)
del_curves["TreeSHAP (GBM, on GBM)"] = deletion_gbm(A["TreeSHAP (GBM)"])
del_curves["Random (GBM)"] = deletion_gbm(A["TreeSHAP (GBM)"], random_order=True)
rc = []
for i in ev:
    order = rng.permutation(256); z = Z[i].copy(); c = [f_prob(z[None])[0]]
    for k in range(20): z[order[k]] = 0.0; c.append(f_prob(z[None])[0])
    rc.append(c)
del_curves["Random (MLP)"] = np.mean(rc, 0)
del_auc = {k: float(np.trapezoid(v, dx=1) / 20) for k, v in del_curves.items()}  # lower = more faithful

# leaky-model attributions: TreeSHAP of m_leak on random-split test binders carrying the motif
cand2 = row_te[(y[row_te] == 1) & has_motif[row_te]]
ev2 = rng.choice(cand2, 80, replace=False)
tsh2 = shap.TreeExplainer(m_leak).shap_values(X[ev2])
rec_leak = recovery(tsh2)
mean_abs_leak = np.abs(tsh2).mean(0); mean_abs_hon = np.abs(tsh).mean(0)
share = lambda v: float(v[MOTIF_IDX].sum() / v.sum())
R["P3"] = dict(recovery_top10=rec, motif_share_honest=share(mean_abs_hon), motif_share_leaky=share(mean_abs_leak),
               recovery_leaky_treeshap=rec_leak, spearman=agree.tolist(), jaccard_top10=jacc.tolist(), names=names,
               deletion_auc=del_auc, deletion_curves={k: v.tolist() for k, v in del_curves.items()})
print("P3", {k: round(v, 3) for k, v in rec.items()}, "leaky", rec_leak, flush=True)

# ---------- 4. Practice 4: expert-in-the-loop proxy ----------
# the planted motif plays the role of the expert's prior (e.g., a JASPAR PWM); features not matching it are flagged
flag = []
for r in A["TreeSHAP (GBM)"]:
    top = np.argsort(-np.abs(r))[:TOPK]
    flag.append(np.mean([t not in MOTIF_IDX for t in top]))
R["P4_proxy"] = dict(mean_fraction_top10_unexplained_by_motif=float(np.mean(flag)))

# ---------- 5. Practice 5: uncertainty ----------
def ece(p, yy, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0.0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(yy[m].mean() - p[m].mean())
    return e
def boot_ece(p, yy, B=500):
    n = len(p); v = []
    for _ in range(B):
        j = rng.integers(0, n, n); v.append(ece(p[j], yy[j]))
    return np.percentile(v, [2.5, 97.5]).tolist()
p_te = m_hon.predict_proba(X[te])[:, 1]; p_cal = m_hon.predict_proba(X[cal])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y[cal]); p_iso = iso.predict(p_te)
pl = LogisticRegression().fit(np.log(p_cal / (1 - p_cal + 1e-9) + 1e-9)[:, None], y[cal])
p_pl = pl.predict_proba(np.log(p_te / (1 - p_te + 1e-9) + 1e-9)[:, None])[:, 1]
p_mlp = m0.predict_proba(Z[te])[:, 1]; p_ens = np.mean([m.predict_proba(Z[te])[:, 1] for m in mlps], 0)
unc = {}
for nm, p in [("GBM raw", p_te), ("GBM isotonic", p_iso), ("GBM Platt", p_pl), ("MLP single", p_mlp), ("MLP 5-ensemble", p_ens)]:
    unc[nm] = dict(auroc=float(roc_auc_score(y[te], p)), ece=float(ece(p, y[te])), ece_ci=boot_ece(p, y[te]),
                   brier=float(brier_score_loss(y[te], p)))
R["P5_calibration"] = unc
# threshold behaviour: "validate everything with predicted P >= 0.8"
thr = 0.8
R["P5_threshold"] = dict(
    raw=dict(n=int((p_te >= thr).sum()), precision=float(y[te][p_te >= thr].mean()) if (p_te >= thr).any() else None),
    isotonic=dict(n=int((p_iso >= thr).sum()), precision=float(y[te][p_iso >= thr].mean()) if (p_iso >= thr).any() else None))
# split conformal (LAC), calibration chromosomes -> test chromosomes
def conformal(pc, yc, pt, alpha=0.1):
    s = 1 - np.where(yc == 1, pc, 1 - pc); n = len(s)
    q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = np.stack([(1 - (1 - pt)) <= q + 0, (1 - pt) <= q], 1)  # class0: 1-(1-p)=p ... see below
    s1 = 1 - pt; s0 = 1 - (1 - pt)  # nonconformity if label=1: 1-p ; label=0: p
    sets = np.stack([s0 <= q, s1 <= q], 1)
    return sets, q
sets, q = conformal(p_iso if False else p_cal, y[cal], p_te)
cov = float(np.mean(sets[np.arange(len(y[te])), y[te]])); size = float(sets.sum(1).mean())
R["P5_conformal"] = dict(alpha=0.1, coverage_chrom_shift=cov, mean_set_size=size,
                         frac_empty=float((sets.sum(1) == 0).mean()), frac_both=float((sets.sum(1) == 2).mean()))
# conformal under exchangeable random split (reference)
r_cal = row_te[: len(row_te) // 2]; r_te = row_te[len(row_te) // 2:]
pc2 = m_leak.predict_proba(X[r_cal])[:, 1]; pt2 = m_leak.predict_proba(X[r_te])[:, 1]
sets2, _ = conformal(pc2, y[r_cal], pt2)
R["P5_conformal"]["coverage_random_split"] = float(np.mean(sets2[np.arange(len(r_te)), y[r_te]]))
# selective prediction: abstain when |p-0.5| small
conf = np.abs(p_iso - 0.5); order = np.argsort(-conf); pred = (p_iso >= 0.5).astype(int)
cov_grid = np.linspace(0.1, 1, 19); acc = []
for c in cov_grid:
    k = int(c * len(order)); sel = order[:k]; acc.append(float((pred[sel] == y[te][sel]).mean()))
R["P5_selective"] = dict(coverage=cov_grid.tolist(), accuracy=acc)
R["timing_s"] = time.time() - t0

# ---------- save ----------
np.save("curve_p_te.npy", p_te); np.save("curve_p_iso.npy", p_iso); np.save("curve_y_te.npy", y[te])
json.dump(R, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["P2", "P3_models", "P4_proxy", "P5_threshold", "P5_conformal"]}, indent=1, default=float))
print(json.dumps(R["P5_calibration"], indent=1))
print(json.dumps({k: R["P3"][k] for k in ["motif_share_honest", "motif_share_leaky", "recovery_leaky_treeshap", "deletion_auc"]}, indent=1))

# ---------- 6. Practice 6: artefacts for the interface recipe ----------
import joblib
joblib.dump({"gbm": m_hon, "iso": iso, "q": float(q), "kmers": KMERS, "motif": MOTIF, "k": K}, "model.joblib")

"""
Orca-1Mb (H1-ESC) for the whole variant table, in batches, with resume.

Per variant (about 44 s): ref + alt + 2 nulls = 4 passes.
  insertion: 2 nulls = insertions of a shuffled copy of the real insert at the SAME position
             (if the insert has too few distinct orders, e.g. a homopolymer: the real insert at flanking positions)
  deletion : 2 nulls = deletions of the same size 200-400 bp upstream / downstream (a deletion cannot be shuffled;
             near_test showed the sensitivity of a site changes by about 5-20% within +-1 kb)
Scores (all relative to the nulls, because the raw size of the change mostly reflects the site and the type):
  rel_<metric>   = variant / mean(null) - 1          (0 = same as the nulls)
  spec_r<R>      = |alt - mean(nulls)| / mean(|null_i - mean(others)|) in a square of +-R bins around the variant;
                   about 1 if alt is as exchangeable with the nulls as they are with each other
  spec_rank_r<R> = rank of the alt residual among the 3 maps (1 = alt deviates most). With no variant-specific
                   effect the ranks are uniform, P(rank 1) = 1/3; this is tested over all variants in summarize()

Run in WSL (env orca_env), from the folder with orca_predict.py. Keep --out OUTSIDE OneDrive:
    CUDA_VISIBLE_DEVICES= python run_all_variants.py --csv sm-project-data/selected_vars_oleksyk2021_orca_subset.csv \
        --out ~/orca_full_out --start 0 --count 50
Next batch: --start 50 --count 50, and so on. Finished variants are skipped.
Logic check without resources: python run_all_variants.py --selftest
"""
import argparse
import os
import sys
import time
from collections import Counter
from math import comb, factorial

import numpy as np
import pandas as pd
import pilot_variant as P

RADII = (6, 12, 25)  # bins of 4 kb: +-24, +-48, +-100 kb
CROP = 50  # bins kept around the variant in the saved .npz maps
KEYS = [
    "local_r6_mean_abs",
    "local_r12_mean_abs",
    "local_r25_mean_abs",
    "coarse4x_local_r25_mean_abs",
    "ins_w10_max_abs_change",
    "one_minus_corr_outside_r25",
]
FIELDS = (
    ["vid", "status", "chrom", "pos", "ref", "alt", "type", "size", "region", "null_type", "null_offsets", "seconds"]
    + [f"{p}_{k}" for p in ("var", "null", "rel") for k in KEYS]
    + [f"spec_r{r}" for r in RADII]
    + [f"spec_rank_r{r}" for r in RADII]
)


def local_sq(M, b0, r):
    lo, hi = max(0, b0 - r), min(M.shape[0], b0 + r + 1)
    return M[lo:hi, lo:hi]


def resid_scores(alt, n0, n1, b0):
    """How far each of the 3 maps is from the mean of the other two, in a square around the variant."""
    out = {}
    for r in RADII:
        A, N0, N1 = (local_sq(x, b0, r) for x in (alt, n0, n1))
        ra = np.nanmean(np.abs(A - (N0 + N1) / 2))
        r0 = np.nanmean(np.abs(N0 - (A + N1) / 2))
        r1 = np.nanmean(np.abs(N1 - (A + N0) / 2))
        m = np.mean([r0, r1])
        out[f"spec_r{r}"] = float(ra / m) if m > 0 else np.nan
        out[f"spec_rank_r{r}"] = int(1 + (r0 > ra) + (r1 > ra)) if m > 0 else np.nan
    return out


def n_distinct_shuffles(seq):
    n = factorial(len(seq))
    for count in Counter(seq).values():
        n //= factorial(count)
    return n


def vid_of(row):
    chrom, pos1, ref, alt = P.parse_row(row)
    return f"{chrom}:{pos1}:{ref}:{alt}"


def process_variant(genome, model, row, mapdir, seed):
    t0 = time.time()
    chrom, pos1, ref, alt = P.parse_row(row)
    k = len(alt) - len(ref)
    kind = "ins" if k > 0 else "del"
    res = {"vid": vid_of(row), "chrom": chrom, "pos": pos1, "ref": ref, "alt": alt, "type": kind,
           "size": abs(k), "region": row.get("Region", "")}

    try:  # MemmapGenome builds len_chrs lazily, on its first query
        genome.get_encoding_from_coords(chrom, 0, 1)
    except Exception:
        pass
    chrlen = getattr(genome, "len_chrs", {}).get(chrom)
    ws = pos1 - 1 - P.HALF
    if chrlen is None or ws < 0 or ws + P.L + abs(k) + 1000 > chrlen:
        res["status"] = "window_outside_chromosome"
        return res
    ok, got = P.check_ref(genome, chrom, pos1, ref)
    if not ok:
        res["status"] = f"ref_mismatch(hg38={got})"
        return res
    ref_enc = genome.get_encoding_from_coords(chrom, ws, ws + P.L)
    too_many_n = (ref_enc.max(axis=1) < 0.5).mean() > 0.01
    if too_many_n or (ref_enc[P.HALF : P.HALF + len(ref)].max(axis=1) < 0.5).any():
        res["status"] = "too_many_N"
        return res

    rng = np.random.default_rng(seed + pos1)

    def predict_at(o, r_len, a_enc):
        return P.predict(model, P.build_alt(genome, chrom, ws, ref_enc, o, r_len, a_enc))

    ref_map = P.predict(model, ref_enc)
    alt_map = predict_at(P.HALF, len(ref), P.enc_str(alt))

    # ---- two nulls ----
    nulls, offs, null_type = [], [], "shuffle"
    real_ins = P.inserted_bases(ref, alt) if kind == "ins" else ""

    def flank_nulls():  # same-size indel 200-400 bp away (the real insert for insertions)
        for d in (-200, 200, -300, 300, -400, 400):
            if len(nulls) == 2:
                break
            o = P.HALF + d
            if (ref_enc[o : o + 1 + abs(k)].max(axis=1) < 0.5).any():
                continue
            if kind == "ins":
                r_len, a_enc = 1, np.concatenate([ref_enc[o : o + 1], P.enc_str(real_ins)], axis=0)
            else:
                r_len, a_enc = 1 + abs(k), ref_enc[o : o + 1]
            nulls.append(predict_at(o, r_len, a_enc))
            offs.append(d)

    shuffles = []
    if kind == "ins" and len(real_ins) == k and n_distinct_shuffles(real_ins) >= 6:
        seen = {real_ins}
        for _ in range(200):
            cand = "".join(rng.permutation(list(real_ins)))
            if cand not in seen:
                seen.add(cand)
                shuffles.append(cand)
            if len(shuffles) == 2:
                break

    if len(shuffles) == 2:  # same position, other order of the inserted bases
        for sh in shuffles:
            a_enc = np.concatenate([ref_enc[P.HALF : P.HALF + 1], P.enc_str(sh)], axis=0)
            nulls.append(predict_at(P.HALF, 1, a_enc))
            offs.append(0)
    else:  # deletions, homopolymers and short repeats
        null_type = "flank"
        flank_nulls()
    if len(nulls) < 2:
        res["status"] = "no_null_positions"
        return res
    res["null_type"] = null_type

    # ---- scores ----
    b0 = P.HALF // P.BIN
    mv = P.metrics(ref_map, alt_map, b0)
    mn = [P.metrics(ref_map, n, (P.HALF + o) // P.BIN) for n, o in zip(nulls, offs)]
    for key in KEYS:
        null_mean = float(np.nanmean([m[key] for m in mn]))
        res[f"var_{key}"], res[f"null_{key}"] = mv[key], null_mean
        res[f"rel_{key}"] = mv[key] / null_mean - 1 if null_mean else np.nan
    res.update(resid_scores(alt_map, nulls[0], nulls[1], b0))
    res["null_offsets"] = ",".join(str(o) for o in offs)

    lo, hi = b0 - CROP, b0 + CROP + 1  # keep the maps around the variant for plotting
    np.savez_compressed(
        os.path.join(mapdir, res["vid"].replace(":", "_")[:150] + ".npz"),
        ref=ref_map[lo:hi, lo:hi], alt=alt_map[lo:hi, lo:hi], n0=nulls[0][lo:hi, lo:hi], n1=nulls[1][lo:hi, lo:hi],
    )
    res["status"] = "ok"
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_batch(genome, model, df, out, start, count, seed=1, log=print):
    out = os.path.expanduser(out)
    mapdir = os.path.join(out, "maps")
    os.makedirs(mapdir, exist_ok=True)
    rpath = os.path.join(out, "results.csv")
    done = set()
    if os.path.exists(rpath):
        old = pd.read_csv(rpath)
        if list(old.columns) != FIELDS:  # file from an older version of the script: add the new columns
            old = old.reindex(columns=FIELDS)
            old.to_csv(rpath, index=False)
        # insertions from the first version are redone (their nulls could have been identical to the variant)
        redo = (old["type"] == "ins") & old["null_type"].isna() & (old["status"] == "ok")
        done = set(old.loc[~redo, "vid"])
    sub = df.iloc[start : start + count]
    t0, n_run = time.time(), 0
    for i, (_, row) in enumerate(sub.iterrows()):
        if vid_of(row) in done:
            continue
        r = process_variant(genome, model, row, mapdir, seed)
        pd.DataFrame([r]).reindex(columns=FIELDS).to_csv(rpath, mode="a", header=not os.path.exists(rpath), index=False)
        n_run += 1
        el = time.time() - t0
        log(f"[{start + i + 1}/{start + len(sub)}] {r['vid']} {r['status']}  elapsed {el / 60:.1f} min, ~{el / n_run * (len(sub) - i - 1) / 60:.0f} min left")
    summarize(rpath, log)


def summarize(rpath, log=print):
    d = pd.read_csv(rpath)
    if "null_type" not in d:
        d["null_type"] = np.nan
    d = d.drop_duplicates("vid", keep="last")
    d.loc[(d["type"] == "del") & d.null_type.isna(), "null_type"] = "flank"
    ok = d[d.status == "ok"]
    log(f"\nresults.csv: {len(d)} variants, {len(ok)} ok; statuses: {d.status.value_counts().to_dict()}")
    log(f"ok by type / null: {ok.groupby(['type', 'null_type']).size().to_dict()}")

    # rank test, only where the null is exchangeable with the variant (same position, shuffled insert)
    sh = ok[ok.null_type == "shuffle"]
    for r in RADII:
        x = sh.dropna(subset=[f"spec_rank_r{r}"])
        n = len(x)
        if n == 0:
            break
        k1 = int((x[f"spec_rank_r{r}"] == 1).sum())
        p = sum(comb(n, j) * (1 / 3) ** j * (2 / 3) ** (n - j) for j in range(k1, n + 1))
        log(f"  rank test (shuffle nulls only) r{r}: alt is the most deviant map in {k1}/{n} = {k1 / n:.2f} "
            f"(0.33 expected without a variant-specific effect), one-sided binomial p = {p:.3g}")

    # relative deviation from the nulls per group; outliers = |robust z| > 3 within the group
    for (t, nt), g in ok.groupby(["type", "null_type"]):
        for key in ("rel_local_r6_mean_abs", "rel_local_r12_mean_abs"):
            v = g[key].dropna()
            if len(v) >= 5:
                log(f"  {t}/{nt} {key}: n={len(v)}, median {v.median():+.3f}, robust SD {1.4826 * (v - v.median()).abs().median():.3f}")
        v = g["rel_local_r6_mean_abs"].dropna()
        if len(v) < 5:
            continue
        med = v.median()
        sd = 1.4826 * (v - med).abs().median()
        if sd > 0:
            z = (g["rel_local_r6_mean_abs"] - med) / sd
            for vid, zz, rel in zip(g.loc[z.abs() > 3, "vid"], z[z.abs() > 3], g.loc[z.abs() > 3, "rel_local_r6_mean_abs"]):
                log(f"    outlier ({t}/{nt}): {vid}  rel_r6 = {rel:+.3f}  ({zz:+.1f} robust SD)")


def selftest():
    import shutil

    g = P._FakeGenome()
    g.len_chrs = {"chr21": len(g.seq)}
    base = lambda p, n: "".join(P.BASES[i] for i in g.seq[p - 1 : p - 1 + n].argmax(axis=1))
    rows = []
    for pos1, kind in [(1_500_000, "ins"), (1_700_000, "del"), (1_900_000, "ins")]:
        r = base(pos1, 12)
        rows.append({"Chr": "21", "Position": pos1, "Ref": r, "Alt": r + "ACGTAC" if kind == "ins" else r[:3], "Region": "intergenic"})
    r = base(1_300_000, 1)
    rows.append({"Chr": "21", "Position": 1_300_000, "Ref": r, "Alt": r + "AAAAAA", "Region": "intergenic"})  # homopolymer insert
    rows.append({"Chr": "21", "Position": 100, "Ref": "A", "Alt": "ACGTACG", "Region": "intergenic"})  # window outside chromosome
    df = pd.DataFrame(rows)
    out = "selftest_all_out"
    shutil.rmtree(out, ignore_errors=True)
    quiet = lambda *a: None
    run_batch(g, P._fake_model, df, out, 0, 10, log=quiet)
    run_batch(g, P._fake_model, df, out, 0, 10, log=quiet)  # resume: nothing new
    res = pd.read_csv(os.path.join(out, "results.csv"))
    assert len(res) == 5 and list(res.status) == ["ok"] * 4 + ["window_outside_chromosome"], res.status
    assert res.loc[3, "null_type"] == "flank" and res.loc[3, "null_offsets"] == "-200,200", res.loc[3]
    print(res[["vid", "status", "type", "null_type", "null_offsets", "rel_local_r6_mean_abs", "spec_rank_r6"]].to_string(index=False))
    summarize(os.path.join(out, "results.csv"))
    shutil.rmtree(out, ignore_errors=True)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sm-project-data/selected_vars_oleksyk2021_orca_subset.csv")
    ap.add_argument("--out", default="~/orca_full_out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    sys.path.append(os.getcwd())
    import torch
    import orca_predict

    torch.set_num_threads(2)
    orca_predict.load_resources(models=["1M"], use_cuda=False)
    df = pd.read_csv(a.csv)
    run_batch(orca_predict.hg38, orca_predict.model_dict_global["h1esc_1m"], df, a.out, a.start, a.count, a.seed)
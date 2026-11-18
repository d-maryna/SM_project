"""
Pilot: effect of one indel on the Orca-1Mb (H1-ESC) prediction, compared with null indels.

Run (WSL, env orca_env, from the folder that contains orca_predict.py):
    CUDA_VISIBLE_DEVICES= python pilot_variant.py --csv sm-project-data/selected_vars_oleksyk2021.csv \
        --chrom chr21 --pos 40686217
Logic check without the model (fake genome and fake model):
    python pilot_variant.py --selftest

Design
- window = 1 Mb, the variant starts at offset 500,000 (bin 125 of 250, bin = 4 kb)
- alt window = ref with the allele swapped; extra flank is taken from hg38 on the right so the length stays 1 Mb
- ref and alt maps are compared bin by bin (a shift of <= 45 bp is far below one 4 kb bin)
- nulls = indels of the same type and size; an inserted sequence is a shuffled copy of the real insert
    "same": at the variant's own position (insertions only)
    "loc" : at random positions within +-100 kb of the centre, at least 20 kb from the variant
- every prediction is cached as .npy, so a re-run continues where it stopped
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

L, HALF, BIN, BASES = 1_000_000, 500_000, 4000, "ACGT"


# ---------- small helpers ----------
def parse_row(row):
    chrom = str(row["Chr"])
    chrom = chrom if chrom.startswith("chr") else "chr" + chrom
    return chrom, int(row["Position"]), str(row["Ref"]), str(row["Alt"])


def inserted_bases(ref, alt):
    return alt[len(ref):] if alt.startswith(ref) else alt[1:]


def enc_str(s):
    a = np.zeros((len(s), 4), dtype=np.float32)
    for i, ch in enumerate(s.upper()):
        if ch in BASES:
            a[i, BASES.index(ch)] = 1.0
    return a


def shuffle_insert(real, k, rng):
    """Same bases in another order (random bases if the allele is unusual)."""
    if len(real) != k:
        return "".join(rng.choice(list(BASES), k))
    sh = real
    for _ in range(20):
        sh = "".join(rng.permutation(list(real)))
        if sh != real:
            break
    return sh


def check_ref(genome, chrom, pos1, ref):
    """pos1 is the 1-based VCF position. True if hg38 matches Ref exactly."""
    e = genome.get_encoding_from_coords(chrom, pos1 - 1, pos1 - 1 + len(ref))
    known = e.max(axis=1) > 0.5
    got = "".join(BASES[i] if ok else "N" for i, ok in zip(e.argmax(axis=1), known))
    return got == ref.upper(), got


def build_alt(genome, chrom, ws, ref_enc, offset, ref_len, alt_enc):
    """Replace ref_enc[offset:offset+ref_len] by alt_enc and refill the right side from hg38 to keep L."""
    right_start = ws + offset + ref_len
    right = genome.get_encoding_from_coords(chrom, right_start, right_start + L - offset - len(alt_enc))
    out = np.concatenate([ref_enc[:offset], alt_enc, right], axis=0)
    assert out.shape == (L, 4), out.shape
    return out


def predict(model, enc):
    if getattr(model, "numpy_input", False):  # only used by --selftest
        return model(enc)
    import torch

    x = torch.FloatTensor(enc[None, :, :]).transpose(1, 2)
    with torch.no_grad():
        return model(x).squeeze().detach().cpu().numpy()


def cached(path, make, log=print):
    if os.path.exists(path):
        return np.load(path)
    t = time.time()
    m = make()
    np.save(path, m)
    log(f"    {os.path.basename(path)} done in {time.time() - t:.0f} s")
    return m


# ---------- metrics on a pair of maps ----------
def nanmean(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def coarsen(a, f):
    n = (a.shape[0] // f) * f
    return a[:n, :n].reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def insulation(M, w):
    out = np.full(M.shape[0], np.nan)
    for i in range(w, len(out) - w):
        out[i] = nanmean(M[i - w : i, i + 1 : i + 1 + w])
    return out


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else np.nan


def metrics(ref, alt, b0, rs=(6, 12, 25), w_ins=10):
    """Size of the change alt - ref around bin b0 (all values are mean |change| unless the name says otherwise)."""
    n = ref.shape[0]
    D = alt - ref
    out = {"global_mean_abs": nanmean(np.abs(D))}
    for r in rs:
        lo, hi = max(0, b0 - r), min(n, b0 + r + 1)
        out[f"local_r{r}_mean_abs"] = nanmean(np.abs(D[lo:hi, lo:hi]))
        out[f"band_r{r}_mean_abs"] = nanmean(np.abs(D[lo:hi, :]))
    r = max(rs)
    lo, hi = max(0, b0 - r), min(n, b0 + r + 1)
    for f in (2, 4, 8):  # bins of 8, 16, 32 kb
        out[f"coarse{f}x_local_r{r}_mean_abs"] = nanmean(np.abs(coarsen(D[lo:hi, lo:hi], f)))
    d = np.abs(insulation(alt, w_ins) - insulation(ref, w_ins))[lo:hi]
    out[f"ins_w{w_ins}_max_abs_change"] = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
    idx = np.arange(n)
    outside = (idx < lo) | (idx >= hi)
    mask = outside[:, None] & outside[None, :]
    out["one_minus_corr_outside_r25"] = 1 - corr(ref[mask], alt[mask])
    out["one_minus_corr_all"] = 1 - corr(ref.ravel(), alt.ravel())
    return out


# ---------- main analysis: variant vs nulls ----------
def run(genome, model, row, outdir, n_null, seed, n_same=0, log=print):
    chrom, pos1, ref, alt = parse_row(row)
    tag = f"{chrom}_{pos1}"
    os.makedirs(outdir, exist_ok=True)

    ok, got = check_ref(genome, chrom, pos1, ref)
    log(f"[{tag}] Ref in file: {ref}   hg38: {got}   match: {ok}")
    if not ok:
        raise SystemExit("Ref does not match hg38 at this position: check coordinates/build.")

    ws = pos1 - 1 - HALF
    ref_enc = genome.get_encoding_from_coords(chrom, ws, ws + L)
    log(f"[{tag}] window {chrom}:{ws}-{ws + L}, fraction of N: {float((ref_enc.max(axis=1) < 0.5).mean()):.4f}")
    k = len(alt) - len(ref)
    kind = "ins" if k > 0 else "del"
    log(f"[{tag}] type: {kind}, size {abs(k)} bp")

    npy = lambda name: os.path.join(outdir, f"{tag}_{name}.npy")
    ref_map = cached(npy("ref"), lambda: predict(model, ref_enc), log)
    alt_enc = build_alt(genome, chrom, ws, ref_enc, HALF, len(ref), enc_str(alt))
    alt_map = cached(npy("alt"), lambda: predict(model, alt_enc), log)
    b0 = HALF // BIN
    obs = metrics(ref_map, alt_map, b0)

    real_ins = inserted_bases(ref, alt) if kind == "ins" else ""

    def make_null(o, rng):
        """(length of ref sequence replaced, new sequence) for a null at offset o, or None if it hits N."""
        anchor = ref_enc[o : o + 1]
        if kind == "del":
            if (ref_enc[o : o + 1 + abs(k)].max(axis=1) < 0.5).any():
                return None
            return 1 + abs(k), anchor
        return 1, np.concatenate([anchor, enc_str(shuffle_insert(real_ins, k, rng))], axis=0)

    def collect(prefix, n, rng, same_pos):
        rows, tries = [], 0
        while len(rows) < n and tries < n * 20:
            tries += 1
            if same_pos:
                o = HALF
            else:
                o = int(rng.integers(HALF - 100_000, HALF + 100_000))
                if abs(o - HALF) < 20_000:
                    continue
            made = make_null(o, rng)
            if made is None:
                continue
            r_len, a_enc = made
            name = f"{prefix}{len(rows):03d}_o{o}"
            nmap = cached(npy(name), lambda: predict(model, build_alt(genome, chrom, ws, ref_enc, o, r_len, a_enc)), log)
            m = metrics(ref_map, nmap, o // BIN)
            m["null_offset"] = o
            rows.append(m)
        return pd.DataFrame(rows)

    nulls = {"loc": collect("loc", n_null, np.random.default_rng(seed), False)}
    if kind == "ins" and n_same > 0:
        nulls["same"] = collect("same", n_same, np.random.default_rng(seed + 1), True)
    for name, nd in nulls.items():
        nd.to_csv(os.path.join(outdir, f"{tag}_null_metrics_{name}.csv"), index=False)

    # variant vs each null set: median, max, z and empirical p = (1 + #null >= variant) / (n + 1)
    res = []
    for metric, v in obs.items():
        r = {"metric": metric, "variant": v}
        for name, nd in nulls.items():
            x = nd[metric].dropna().values
            r[f"{name}_median"] = float(np.median(x)) if len(x) else np.nan
            r[f"{name}_max"] = float(np.max(x)) if len(x) else np.nan
            r[f"{name}_z"] = (v - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else np.nan
            r[f"{name}_p"] = (1 + int((x >= v).sum())) / (len(x) + 1) if len(x) else np.nan
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(outdir, f"{tag}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    log(res.round(5).to_string(index=False))
    for name, nd in nulls.items():
        log(f"null '{name}': n={len(nd)}, smallest possible p = {1 / (len(nd) + 1):.3f}")
    return res


# ---------- is the window centre special? ----------
def center_test(genome, model, row, outdir, deltas, n_shuf, seed, log=print):
    """Insert a shuffled copy of the real insert at the centre of windows built around other sites
    (variant position + delta) and see how big the change is there."""
    chrom, pos1, ref, alt = parse_row(row)
    tag = f"{chrom}_{pos1}"
    k = len(alt) - len(ref)
    if k <= 0:
        raise SystemExit("center_test is written for insertions")
    real_ins = inserted_bases(ref, alt)
    rng = np.random.default_rng(seed + 100)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for d in deltas:
        ws = pos1 - 1 + d - HALF
        ref_enc = genome.get_encoding_from_coords(chrom, ws, ws + L)
        if (ref_enc[HALF : HALF + 1].max(axis=1) < 0.5).any() or (ref_enc.max(axis=1) < 0.5).mean() > 0.01:
            log(f"  delta {d:+d}: N in window, skipped")
            continue
        rmap = cached(os.path.join(outdir, f"{tag}_ctr{d:+d}_ref.npy"), lambda: predict(model, ref_enc), log)
        for i in range(n_shuf):
            a_enc = np.concatenate([ref_enc[HALF : HALF + 1], enc_str(shuffle_insert(real_ins, k, rng))], axis=0)
            amap = cached(
                os.path.join(outdir, f"{tag}_ctr{d:+d}_s{i}.npy"),
                lambda: predict(model, build_alt(genome, chrom, ws, ref_enc, HALF, 1, a_enc)),
                log,
            )
            m = metrics(rmap, amap, HALF // BIN)
            rows.append({"delta_bp": d, "shuffle": i, **{x: m[x] for x in ("local_r6_mean_abs", "local_r12_mean_abs", "local_r25_mean_abs", "global_mean_abs")}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"{tag}_center_test.csv"), index=False)
    pd.set_option("display.width", 200)
    log(df.round(5).to_string(index=False))
    sp = os.path.join(outdir, f"{tag}_summary.csv")
    if os.path.exists(sp):
        sm = pd.read_csv(sp).set_index("metric")
        log(f"for comparison, the real variant (earlier run): local_r6 = {sm.loc['local_r6_mean_abs', 'variant']:.5f}, local_r12 = {sm.loc['local_r12_mean_abs', 'variant']:.5f}")
    log("random positions (earlier run) had local_r6 about 0.003-0.009")
    return df


# ---------- how fast does sensitivity change with distance? ----------
def near_test(genome, model, row, outdir, deltas, seed, log=print):
    """In the variant's own window put a shuffled-insert and a same-size deletion at +-delta from the variant."""
    chrom, pos1, ref, alt = parse_row(row)
    tag = f"{chrom}_{pos1}"
    k = abs(len(alt) - len(ref))
    real_ins = inserted_bases(ref, alt) if len(alt) > len(ref) else ""
    rng = np.random.default_rng(seed + 200)
    ws = pos1 - 1 - HALF
    ref_enc = genome.get_encoding_from_coords(chrom, ws, ws + L)
    ref_path = os.path.join(outdir, f"{tag}_ref.npy")
    if not os.path.exists(ref_path):
        raise SystemExit(f"{ref_path} not found: run the main pilot first")
    rmap = np.load(ref_path)
    rows = []
    for kind in ("ins", "del"):
        for d in deltas:
            o = HALF + d
            if kind == "del":
                if (ref_enc[o : o + 1 + k].max(axis=1) < 0.5).any():
                    continue
                r_len, a_enc = 1 + k, ref_enc[o : o + 1]
            else:
                r_len, a_enc = 1, np.concatenate([ref_enc[o : o + 1], enc_str(shuffle_insert(real_ins, k, rng))], axis=0)
            amap = cached(
                os.path.join(outdir, f"{tag}_near_{kind}{d:+d}.npy"),
                lambda: predict(model, build_alt(genome, chrom, ws, ref_enc, o, r_len, a_enc)),
                log,
            )
            m = metrics(rmap, amap, o // BIN)
            rows.append({"type": kind, "offset_bp": d, **{x: m[x] for x in ("local_r6_mean_abs", "local_r12_mean_abs", "local_r25_mean_abs")}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"{tag}_near_test.csv"), index=False)
    pd.set_option("display.width", 200)
    log(df.round(5).to_string(index=False))
    log("for comparison: the real insertion at offset 0 had local_r6 = 0.01487; same-position shuffles ~0.0152")
    return df


# ---------- self-test with a fake genome and model ----------
class _FakeGenome:
    def __init__(self, n=3_000_000, seed=1):
        r = np.random.default_rng(seed)
        self.seq = np.zeros((n, 4), dtype=np.float32)
        self.seq[np.arange(n), r.integers(0, 4, n)] = 1.0

    def get_encoding_from_coords(self, chrom, start, end, strand="+", pad=False):
        assert 0 <= start and end <= len(self.seq)
        return self.seq[start:end].copy()


def _fake_model(enc):
    r = np.random.default_rng(int(enc[:, 0].sum() * 7 + enc[::1000, 1].sum()) % 2**31)
    m = r.random((250, 250))
    return (m + m.T) / 2


_fake_model.numpy_input = True


def selftest():
    import shutil

    g = _FakeGenome()
    pos1 = 1_500_000
    ref = "".join(BASES[i] for i in g.seq[pos1 - 1 : pos1 - 1 + 12].argmax(axis=1))
    for alt, label in [(ref + "ACGTA", "insertion"), (ref[:2], "deletion")]:
        print(f"--- selftest {label} ---")
        run(g, _fake_model, {"Chr": "21", "Position": pos1, "Ref": ref, "Alt": alt}, "selftest_out", n_null=4, seed=1, n_same=3)
    row = {"Chr": "21", "Position": pos1, "Ref": ref, "Alt": ref + "ACGTA"}
    print("--- selftest center_test ---")
    center_test(g, _fake_model, row, "selftest_out", [-150_000, 50_000], 2, 1, log=lambda *a: None)
    print("--- selftest near_test ---")
    near_test(g, _fake_model, row, "selftest_out", [-1000, 0, 1000], 1, log=lambda *a: None)
    shutil.rmtree("selftest_out", ignore_errors=True)
    print("selftest OK: lengths, Ref check, resume and metrics ran without errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sm-project-data/selected_vars_oleksyk2021.csv")
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--pos", type=int, default=40686217)
    ap.add_argument("--n_null", type=int, default=60, help="random-position nulls")
    ap.add_argument("--n_same", type=int, default=20, help="same-position nulls (insertions only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="pilot_out2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--near_test", action="store_true", help="insertions and deletions at small offsets from the variant")
    ap.add_argument("--center_test", action="store_true", help="same-size insertion at the centre of windows around other sites")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)

    sys.path.append(os.getcwd())
    import torch
    import orca_predict

    torch.set_num_threads(2)
    orca_predict.load_resources(models=["1M"], use_cuda=False)
    genome, model = orca_predict.hg38, orca_predict.model_dict_global["h1esc_1m"]

    df = pd.read_csv(a.csv)
    chroms = df["Chr"].astype(str).map(lambda c: c if c.startswith("chr") else "chr" + c)
    sel = df[(chroms == a.chrom) & (df["Position"] == a.pos)]
    if sel.empty:
        raise SystemExit(f"{a.chrom}:{a.pos} not found in {a.csv}")
    row, t0 = sel.iloc[0], time.time()
    if a.near_test:
        near_test(genome, model, row, a.out, [-5000, -1000, -200, 0, 200, 1000, 5000], a.seed)
    elif a.center_test:
        center_test(genome, model, row, a.out, [-150_000, -100_000, -50_000, 50_000, 100_000, 150_000], 2, a.seed)
    else:
        run(genome, model, row, a.out, a.n_null, a.seed, a.n_same)
    print(f"total time: {(time.time() - t0) / 60:.1f} min")
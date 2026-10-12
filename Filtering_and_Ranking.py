import argparse, json, math, re, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

try:                                   # scipy là tuỳ chọn; có thì nhanh hơn và tốn ít RAM hơn
    from scipy.spatial import cKDTree
except ImportError:                    # pragma: no cover
    cKDTree = None

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())
DNA = {"DA", "DC", "DG", "DT", "DU", "DI", "DN"}
RNA = {"A", "C", "G", "U", "I", "N"}
GLYCAN = set("NAG NDG MAN BMA FUC FUL GAL GLA BGC GLC SIA XYS XYP A2G NGA BDP RAM".split())
POLYMER = {"protein", "dna", "rna"}
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
         "CA": 2.31, "NA": 2.27, "MG": 1.73, "ZN": 1.39, "FE": 1.94, "K": 2.75}
PROBE = 1.4
TOKEN = re.compile(r'''"(?:[^"]|"(?!\s))*"(?=\s|$)|'(?:[^']|'(?!\s))*'(?=\s|$)|\S+''')
UPLOAD_PREFIX = re.compile(r"^\d{10,}_")
REQUIRED_CSV = ["sequence", "ptm", "iptm", "fraction_disordered", "has_clash", "ranking_score",
                "pair_iptm_mean", "pair_iptm_min", "pair_pae_min_max"]

#
def _sphere(n=96):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0, 1 - z * z))
    a = math.pi * (1 + math.sqrt(5)) * i
    return np.column_stack((r * np.cos(a), r * np.sin(a), z))

SPH = _sphere()

# đọc file .CIF
def _tokens(line):
    return [t[1:-1] if len(t) > 1 and t[0] in "'\"" and t[-1] == t[0] else t
            for t in TOKEN.findall(line)]


def read_category(lines, prefix):
    """Đọc 1 category mmCIF (dạng loop_ hoặc dạng key-value một dòng). Trả về (tags, rows)."""
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s == "loop_":
            j, tags = i + 1, []
            while j < n and lines[j].startswith("_"):
                tags.append(lines[j].split()[0]); j += 1
            if tags and tags[0].startswith(prefix):
                rows, buf = [], []
                while j < n:
                    t = lines[j].strip()
                    if not t:
                        j += 1; continue
                    if t.startswith(("#", "loop_", "_", "data_")):
                        break
                    buf += _tokens(t)
                    while len(buf) >= len(tags):
                        rows.append(buf[:len(tags)]); buf = buf[len(tags):]
                    j += 1
                return tags, rows
            i = j
        elif s.startswith(prefix):        # category chỉ có 1 dòng -> không có loop_
            d = {}
            while i < n and lines[i].startswith(prefix):
                tok = _tokens(lines[i].strip())
                if len(tok) >= 2:
                    d[tok[0]] = tok[1]
                i += 1
            return list(d), [list(d.values())]
        else:
            i += 1
    return [], []


def parse_cif(path):
    """Trả về (DataFrame nguyên tử nặng, tập liên kết cộng hoá trị từ _struct_conn)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    tags, rows = read_category(lines, "_atom_site.")
    if not rows:
        raise ValueError(f"No _atom_site loop in {path}")
    col = {t: k for k, t in enumerate(tags)}
    need = ["group_PDB", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id",
            "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv"]
    miss = [n for n in need if "_atom_site." + n not in col]
    if miss:
        raise ValueError(f"{path}: missing _atom_site columns {miss}")
    g = lambda n: [r[col["_atom_site." + n]] for r in rows]
    df = pd.DataFrame({
        "group": g("group_PDB"), "element": [e.upper() for e in g("type_symbol")],
        "atom": g("label_atom_id"), "comp": g("label_comp_id"), "chain": g("label_asym_id"),
        "seq": g("label_seq_id"),
        "x": np.array(g("Cartn_x"), float), "y": np.array(g("Cartn_y"), float),
        "z": np.array(g("Cartn_z"), float), "plddt": np.array(g("B_iso_or_equiv"), float)})
    df = df[~df.element.isin(["H", "D"])].reset_index(drop=True)
    df["res"] = df.chain + ":" + df.seq.where(df.seq != ".", "0")
    df["key"] = df.chain + "|" + df.seq + "|" + df.atom       # định danh nguyên tử, khớp với _struct_conn

    # liên kết cộng hoá trị (glycan-protein, disulfide, ligand liên kết...) KHÔNG phải clash
    bonds = set()
    ctags, crows = read_category(lines, "_struct_conn.")
    if crows:
        cc = {t.split(".", 1)[1]: k for k, t in enumerate(ctags)}
        try:
            for r in crows:
                k = []
                for p in ("ptnr1", "ptnr2"):
                    k.append(f"{r[cc[p + '_label_asym_id']]}|{r[cc[p + '_label_seq_id']]}|{r[cc[p + '_label_atom_id']]}")
                bonds.add((k[0], k[1])); bonds.add((k[1], k[0]))
        except KeyError:
            bonds = set()
    return df, bonds


def classify_chains(df):
    info = {}
    for ch, g in df.groupby("chain", sort=False):
        comps = set(g.comp)
        if comps <= AA: t = "protein"
        elif comps <= DNA: t = "dna"
        elif comps <= RNA and (g.group == "ATOM").all(): t = "rna"
        elif len(g) == 1: t = "ion"
        elif comps <= GLYCAN: t = "glycan"
        else: t = "ligand"
        info[ch] = t
    return info


# Xử lý hình học
def close_pairs(xa, xb, cut):
    """Mọi cặp nguyên tử (a_i, b_j) có khoảng cách < cut -> (i, j, d)."""
    if cKDTree is not None:
        m = cKDTree(xa).sparse_distance_matrix(cKDTree(xb), cut, output_type="ndarray")
        return m["i"].astype(int), m["j"].astype(int), np.asarray(m["v"], float)
    d2 = np.maximum((xa * xa).sum(1)[:, None] + (xb * xb).sum(1)[None, :] - 2 * xa @ xb.T, 0)
    i, j = np.where(d2 < cut * cut)
    return i, j, np.sqrt(d2[i, j])


def _sasa_atoms(q_xyz, q_r, o_xyz, o_r):
    total = 0.0
    for i in range(len(q_xyz)):
        d2 = ((o_xyz - q_xyz[i]) ** 2).sum(1)
        m = (d2 < (q_r[i] + o_r) ** 2) & (d2 > 1e-8)
        if m.any():
            pts = q_xyz[i] + SPH * q_r[i]
            dd = ((pts[:, None, :] - o_xyz[m][None]) ** 2).sum(-1)
            frac = (dd >= o_r[m] ** 2).all(1).mean()
        else:
            frac = 1.0
        total += 4 * math.pi * q_r[i] ** 2 * frac
    return total


def buried_sasa(xa, ra, xb, rb, qa, qb):
    xab, rab = np.vstack([xa, xb]), np.concatenate([ra, rb])
    res = 0.0
    for q, x, r in ((qa, xa, ra), (qb, xb, rb)):
        res += _sasa_atoms(x[q], r[q], x, r) - _sasa_atoms(x[q], r[q], xab, rab)
    return max(0.0, res)


def _ipsae_dir(sub, cutoff=10.0):
    valid = sub < cutoff
    cnt = valid.sum(1)
    d0 = np.maximum(1.0, 1.24 * (np.maximum(cnt, 27) - 15) ** (1 / 3) - 1.8)
    tm = 1 / (1 + (sub / d0[:, None]) ** 2)
    per = (tm * valid).sum(1) / np.maximum(cnt, 1)
    return float(per.max()) if len(per) else 0.0


def ipsae_pair(pae_json, c1, c2):
    d = json.loads(Path(pae_json).read_text())
    tok = np.asarray(d["token_chain_ids"]); pae = np.asarray(d["pae"], float)
    i1, i2 = np.flatnonzero(tok == c1), np.flatnonzero(tok == c2)
    if not len(i1) or not len(i2):
        return np.nan
    return max(_ipsae_dir(pae[np.ix_(i1, i2)]), _ipsae_dir(pae[np.ix_(i2, i1)]))


# phân tích 1 mô hình
def analyse_model(cif, a, full_data=None, summary=None):
    df, bonds = parse_cif(cif)
    types = classify_chains(df)
    sub = {c: g.reset_index(drop=True) for c, g in df.groupby("chain", sort=False)}
    xyz = {c: g[["x", "y", "z"]].to_numpy() for c, g in sub.items()}
    rad = {c: np.array([RADII.get(e, 1.7) + PROBE for e in g.element]) for c, g in sub.items()}
    chains = list(types)
    pairs, n_bond_excl = [], 0
    for i, c1 in enumerate(chains):
        for c2 in chains[i + 1:]:
            t1, t2 = types[c1], types[c2]
            if "ion" in (t1, t2) or not ({t1, t2} & POLYMER):
                continue
            if {t1, t2} <= {"dna", "rna"} and not a.keep_na_na:
                continue                      # duplex base-pairing không phải interface gắn kết
            g1, g2 = sub[c1], sub[c2]
            reach = max(a.contact, rad[c1].max() + rad[c2].max())
            ia, ib, d = close_pairs(xyz[c1], xyz[c2], reach)
            ct = d < a.contact
            if not ct.any():
                continue
            ra_, rb_ = g1.res.to_numpy(), g2.res.to_numpy()
            r1, r2 = set(ra_[ia[ct]]), set(rb_[ib[ct]])
            res_pairs = set(zip(ra_[ia[ct]], rb_[ib[ct]]))
            pl = np.concatenate([g1.plddt[g1.res.isin(r1)], g2.plddt[g2.res.isin(r2)]])
            no1 = g1.element.isin(["N", "O"]).to_numpy(); no2 = g2.element.isin(["N", "O"]).to_numpy()
            polar = int((no1[ia] & no2[ib] & (d < 3.5)).sum())
            # clash hình học: cặp nguyên tử quá gần, TRỪ các cặp có liên kết cộng hoá trị
            k1, k2 = g1.key.to_numpy(), g2.key.to_numpy()
            clash_idx = np.flatnonzero(d < a.clash_dist)
            clash = 0
            for t in clash_idx:
                if (k1[ia[t]], k2[ib[t]]) in bonds:
                    n_bond_excl += 1
                else:
                    clash += 1
            pairs.append({
                "chain_pair": f"{c1}_{c2}", "types": f"{t1}-{t2}",
                "contact_atom_pairs": int(ct.sum()), "contact_residue_pairs": len(res_pairs),
                "interface_residues": len(r1) + len(r2), "polar_contacts": polar,
                "iface_plddt": float(pl.mean()), "frac_iface_plddt_lt70": float((pl < 70).mean()),
                "min_dist": float(d.min()), "clash_pairs": clash,
                "buried_sasa": buried_sasa(xyz[c1], rad[c1], xyz[c2], rad[c2],
                                           np.unique(ia), np.unique(ib)),
                "ipsae": ipsae_pair(full_data, c1, c2) if full_data else np.nan})

    # ion: coi là "bám" nếu có nguyên tử không-phải-ion trong vòng ion_dist
    is_ion = df.chain.map(types).eq("ion").to_numpy()
    ions, nonion = df[is_ion], df[~is_ion]
    unbound = 0
    if len(ions) and len(nonion):
        ix, nx = ions[["x", "y", "z"]].to_numpy(), nonion[["x", "y", "z"]].to_numpy()
        hit = np.zeros(len(ix), bool)
        i_, _, _ = close_pairs(ix, nx, a.ion_dist)
        hit[np.unique(i_)] = True
        unbound = int((~hit).sum())

    m = {"cif_used": str(cif), "chain_types": ",".join(f"{c}:{t}" for c, t in types.items()),
         "global_plddt": float(df.plddt.mean()), "frac_plddt_lt50": float((df.plddt < 50).mean()),
         "n_ions": len(ions), "n_ions_unbound": unbound, "n_primary_interfaces": len(pairs),
         "n_covalent_pairs_excluded": n_bond_excl}
    if pairs:
        pdf = pd.DataFrame(pairs)
        best = pdf.loc[pdf.buried_sasa.idxmax()]
        w = pdf.contact_atom_pairs
        m.update({
            "best_pair": best.chain_pair, "best_pair_type": best.types,
            "bsasa_best": best.buried_sasa, "bsasa_total": pdf.buried_sasa.sum(),
            "contact_res_pairs_best": best.contact_residue_pairs,
            "contact_atoms_total": int(w.sum()), "polar_contacts_total": int(pdf.polar_contacts.sum()),
            "iface_plddt": float((pdf.iface_plddt * w).sum() / w.sum()),
            "min_dist_primary": float(pdf.min_dist.min()),
            "clash_pairs_total": int(pdf.clash_pairs.sum()),
            "ipsae_max": pdf.ipsae.max() if pdf.ipsae.notna().any() else np.nan})
    if summary:   # tính lại iPTM / PAE chỉ trên cặp interface chính (bỏ ion, đường-đường)
        try:
            js = json.loads(Path(summary).read_text())
            M, P = np.asarray(js["chain_pair_iptm"], float), np.asarray(js["chain_pair_pae_min"], float)
            if M.shape[0] != len(chains):
                raise ValueError(f"matrix {M.shape[0]} chains != CIF {len(chains)} chains")
            idx = {c: k for k, c in enumerate(chains)}
            vals, paes = [], []
            for r in pairs:
                c1, c2 = r["chain_pair"].split("_")
                vals.append((M[idx[c1], idx[c2]] + M[idx[c2], idx[c1]]) / 2)
                paes.append(max(P[idx[c1], idx[c2]], P[idx[c2], idx[c1]]))
            if vals:
                m.update({"prim_pair_iptm_mean": float(np.mean(vals)),
                          "prim_pair_iptm_min": float(np.min(vals)),
                          "prim_pair_pae_min_max": float(np.max(paes))})
        except (KeyError, IndexError, ValueError) as e:
            m["summary_error"] = repr(e)
    return m, pairs


def _job(args):
    cif, a, fd, sm, key = args
    try:
        m, prs = analyse_model(cif, a, fd, sm)
        return key, {"status": "ok", **m}, prs
    except Exception as e:  # noqa: BLE001
        return key, {"status": f"error: {e!r}"}, []


# Tầng 1 và Tầng 2 (từ CSV)
def _clip01(x):
    return x.clip(0, 1)


def _reasons(frame, rules):
    out = np.array([""] * len(frame), dtype=object)
    for _, bad, label in rules:
        txt = label(frame) if callable(label) else label
        txt = np.asarray(txt if isinstance(txt, pd.Series) else [txt] * len(frame), dtype=object)
        bad = bad.fillna(False).to_numpy(bool)
        out = np.where(bad, np.where(out == "", txt, out + ";" + txt), out)
    return pd.Series(out, index=frame.index, dtype=object)


def tier1(df, a):
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    ptm, iptm, dis, clash = num("ptm"), num("iptm"), num("fraction_disordered"), num("has_clash")
    rules = [
        ("clash", clash.isna() | (clash > 0.5), "has_clash"),
        ("ptm", ptm.isna() | (ptm < a.min_ptm), "ptm<%.2f" % a.min_ptm),
        ("iptm", iptm.isna() | (iptm < a.min_iptm), "iptm<%.2f" % a.min_iptm),
        ("dis", dis.isna() | (dis > a.max_disordered), "disordered>%.2f" % a.max_disordered)]
    df["t1_fail"] = _reasons(df, rules)
    df["tier1_pass"] = df.t1_fail == ""
    return df


def tier2(df, a):
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    rk, im, imin, pae = num("ranking_score"), num("pair_iptm_mean"), num("pair_iptm_min"), num("pair_pae_min_max")
    w = np.array([a.w_rank, a.w_iptm_mean, a.w_pae, a.w_iptm_min], float)
    w = w / w.sum()
    parts = [_clip01(rk), _clip01(im), _clip01(1 - pae / a.max_pair_pae), _clip01(imin)]
    df["tier2_score"] = sum(wi * p for wi, p in zip(w, parts))
    rules = [
        ("pae", pae.isna() | (pae > a.max_pair_pae), lambda f: "pair_pae>%g" % a.max_pair_pae),
        ("iptm", im.isna() | (im < a.min_pair_iptm_mean), lambda f: "pair_iptm_mean<%g" % a.min_pair_iptm_mean)]
    df["t2_fail"] = _reasons(df, rules)
    df["tier2_pass"] = df.tier1_pass & (df.t2_fail == "")
    df.loc[~df.tier1_pass, "tier2_score"] = np.nan
    return df


# điểm cấu trúc + Tầng 3
def _s(x, lo, hi):
    return ((x - lo) / (hi - lo)).clip(0, 1)


def structure_score(df):
    parts = {
        "iface_plddt": (0.40, _s(df["iface_plddt"], 60, 95)),
        "bsasa": (0.30, _s(np.log10(df["bsasa_total"].clip(lower=1)), math.log10(200), math.log10(6000))),
        "contacts": (0.15, _s(df["contact_res_pairs_best"], 5, 80)),
        "ipsae": (0.15, _s(df.get("ipsae_max", pd.Series(np.nan, index=df.index)), 0.2, 0.8))}
    num = sum(w * s.fillna(0) for w, s in parts.values())
    den = sum(w * s.notna() for w, s in parts.values())
    return num / den.replace(0, np.nan)


def infer_sample(df):
    if "sample" in df:
        return df["sample"].astype(int)
    s = pd.Series(np.nan, index=df.index)
    if "source_file" in df:
        s = s.fillna(df.source_file.astype(str).str.extract(r"_(\d+)\.json$")[0].astype(float))
    if "cif_file" in df:
        s = s.fillna(df.cif_file.astype(str).str.extract(r"_model_(\d+)\.cif$")[0].astype(float))
    if "candidate" in df:
        s = s.fillna(df.candidate.astype(float) - 1)
    if s.isna().any():
        raise SystemExit("Không suy ra được chỉ số 'sample'")
    return s.astype(int)


def index_files(cif_dir):
    idx = {}
    for f in Path(cif_dir).rglob("*"):
        if f.is_file():
            idx.setdefault(UPLOAD_PREFIX.sub("", f.name), []).append(f)
    return idx


def lookup(idx, name):
    hit = idx.get(name, [])
    if len(hit) > 1:
        raise ValueError(f"Ambiguous match for {name}: {[str(h) for h in hit[:3]]}")
    return hit[0] if hit else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="CSV ứng viên")
    p.add_argument("--cif-dir", required=True)
    p.add_argument("--out", default="funnel_out")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--copy-cifs", action="store_true")
    g = p.add_argument_group("Tầng 1")
    g.add_argument("--min-ptm", type=float, default=0.60)
    g.add_argument("--min-iptm", type=float, default=0.50)
    g.add_argument("--max-disordered", type=float, default=0.20)
    g = p.add_argument_group("Tầng 2")
    g.add_argument("--max-pair-pae", type=float, default=8.0)
    g.add_argument("--min-pair-iptm-mean", type=float, default=0.5)
    g.add_argument("--w-rank", type=float, default=0.30)
    g.add_argument("--w-iptm-mean", type=float, default=0.35)
    g.add_argument("--w-pae", type=float, default=0.20)
    g.add_argument("--w-iptm-min", type=float, default=0.15)
    g = p.add_argument_group("Cấu trúc từ CIF")
    g.add_argument("--contact", type=float, default=5.0)
    g.add_argument("--clash-dist", type=float, default=2.0)
    g.add_argument("--max-clash-pairs", type=int, default=3)
    g.add_argument("--ion-dist", type=float, default=3.3)
    g.add_argument("--keep-na-na", action="store_true")
    g.add_argument("--min-iface-plddt", type=float, default=70.0)
    g.add_argument("--min-bsasa", type=float, default=200.0)
    g.add_argument("--min-ipsae", type=float, default=0.3)
    g.add_argument("--min-prim-iptm", type=float, default=0.5)
    g.add_argument("--max-unbound-ions", type=int, default=None)
    g = p.add_argument_group("Tầng 3")
    g.add_argument("--per-seq", type=int, default=3)
    g.add_argument("--w-tier2", type=float, default=0.5)
    g.add_argument("--n3", type=int, default=0)

    if any("ipykernel" in arg or "colab_kernel_launcher" in arg for arg in sys.argv):
        # Thiết lập các tham số mặc định từ file CSV hiện có của bạn
        a = p.parse_args(args=["--cif-dir", ".", "--input", "af3_candidates.csv"])
    else:
        a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = index_files(a.cif_dir)

    # ---------------------------------------------------------------- nạp dữ liệu
    base = pd.read_csv(a.input, encoding="utf-8-sig")
    base.columns = [c.strip() for c in base.columns]
    base["sample"] = infer_sample(base)
    if "cif_file" in base:
        base = base.rename(columns={"cif_file": "cif_file_csv"})
    if base.duplicated(["sequence", "sample"]).any():
        base = base.drop_duplicates(["sequence", "sample"])
    base = tier2(tier1(base.reset_index(drop=True), a), a)
    to_cif = base[base.tier2_pass]

    #  phân tích CIF (chỉ xử lý ngầm, loại bỏ in log ok/lỗi)
    jobs = []
    for r in to_cif.itertuples(index=False):
        seq, smp = r.sequence, int(r.sample)
        cif = lookup(idx, f"{seq}_model_{smp}.cif")
        if cif is None: continue
        jobs.append((cif, a, lookup(idx, f"{seq}_full_data_{smp}.json"),
                     lookup(idx, f"{seq}_summary_confidences_{smp}.json"), (seq, smp)))
    
    if jobs and a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]

    metrics, pair_rows = [], []
    for (seq, smp), m, prs in results:
        key = {"sequence": seq, "sample": smp}
        metrics.append({**key, **m})
        pair_rows += [{**key, **x} for x in prs]

    met = pd.DataFrame(metrics) if metrics else pd.DataFrame(columns=["sequence", "sample", "status"])
    met.to_csv(out / "stage3_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(out / "stage3_pairs.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- lọc cấu trúc + Tầng 3
    allm = base.merge(met, on=["sequence", "sample"], how="left")
    st = allm.status.fillna("")
    ok = st.eq("ok")
    num = lambda c: pd.to_numeric(allm[c], errors="coerce") if c in allm else pd.Series(np.nan, index=allm.index)
    rules = [
        ("cif", allm.tier2_pass & ~ok, lambda f: "cif:" + st.where(st != "", "not_run")),
        ("iface", ok & (num("n_primary_interfaces").fillna(0) == 0), "no_interface"),
        ("clash", ok & (num("clash_pairs_total") > a.max_clash_pairs), "geom_clash>%d" % a.max_clash_pairs),
        ("plddt", ok & (num("iface_plddt") < a.min_iface_plddt), "iface_plddt<%g" % a.min_iface_plddt),
        ("bsasa", ok & (num("bsasa_total") < a.min_bsasa), "bsasa<%g" % a.min_bsasa),
        ("ipsae", ok & (num("ipsae_max") < a.min_ipsae), "ipsae<%g" % a.min_ipsae),
        ("iptm", ok & (num("prim_pair_iptm_mean") < a.min_prim_iptm), "prim_iptm<%g" % a.min_prim_iptm)]
    
    allm["t3_fail"] = _reasons(allm, rules)
    allm.loc[~allm.tier2_pass, "t3_fail"] = ""
    allm["struct_pass"] = allm.tier2_pass & (allm.t3_fail == "")

    cand = allm[allm.struct_pass].copy()
    if len(cand):
        cand["struct_score"] = structure_score(cand)
        t2 = cand.tier2_score
        cand["final_score"] = np.where(t2.notna(), a.w_tier2 * t2.fillna(0) + (1 - a.w_tier2) * cand.struct_score,
                                       cand.struct_score)
        cand = cand.sort_values("final_score", ascending=False)
        cand["rank_in_seq"] = cand.groupby("sequence", sort=False).cumcount() + 1
        final = cand[cand.rank_in_seq <= a.per_seq]
        if a.n3 > 0:
            final = final.head(a.n3)
        final = final.copy()
        final.insert(0, "final_rank", range(1, len(final) + 1))
        allm = allm.merge(final[["sequence", "sample", "final_rank", "struct_score", "final_score"]], 
                          on=["sequence", "sample"], how="left")
        allm["final_rank"] = allm.final_rank.astype("Int64")
        allm["tier3_pass"] = allm.final_rank.notna()
    else:
        allm["final_rank"] = pd.array([pd.NA] * len(allm), dtype="Int64")
        allm["tier3_pass"] = False

    allm["stage_reached"] = np.select(
        [allm.tier3_pass, allm.struct_pass, allm.tier2_pass, allm.tier1_pass],
        ["FINAL", "tier3_capped", "tier2_ok_struct_fail", "tier1_ok_tier2_fail"], "tier1_fail")
    allm["fail_reason"] = (allm.t1_fail + ";" + allm.t2_fail.where(allm.tier1_pass, "") + ";" + allm.t3_fail)\
        .str.replace(r";+", ";", regex=True).str.strip(";")
        
    # Lưu tất cả các file CSV cần thiết
    allm.to_csv(out / "funnel_all_models.csv", index=False, encoding="utf-8-sig")
    allm[allm.tier1_pass].to_csv(out / "tier1_pass.csv", index=False, encoding="utf-8-sig")
    allm[allm.tier2_pass].to_csv(out / "tier2_pass.csv", index=False, encoding="utf-8-sig")
    
    final = allm[allm.tier3_pass].sort_values("final_rank")
    final.to_csv(out / "tier3_final.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- Chỉ in kết quả cuối
    if len(final):
        cols = [c for c in ["final_rank", "sequence", "sample", "best_pair_type", "tier2_score", "struct_score",
                            "final_score", "bsasa_total", "iface_plddt", "clash_pairs_total", "n_ions_unbound"]
                if c in final]
        print("=== KẾT QUẢ LỌC CUỐI CÙNG (TIER 3 FINAL) ===")
        print(final[cols].round(3).to_string(index=False))
        print(f"\n[Xong] Đã xuất {len(final)} ứng viên đạt chuẩn vào file '{out}/tier3_final.csv'")
    else:
        print("Không có ứng viên nào đạt tất cả tiêu chuẩn lọc.")

if __name__ == "__main__":
    main()
import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCALAR_KEYS = ["ranking_score", "ptm", "iptm", "fraction_disordered", "has_clash", "num_recycles"]


def clean(v):
    """None/NaN -> chuỗi rỗng, làm tròn số cho gọn."""
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else round(v, 4)
    return v


def stats(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return "", ""
    return round(min(vals), 4), round(sum(vals) / len(vals), 4)


def parse_file_info(path: Path):
    """
    Tách thông tin sequence và candidate index.
    - sequence: Tên thư mục hoặc tiền tố của chuỗi (ví dụ: fold_protein_rna_ion_pdb_8aw3)
    - candidate: Số thứ tự mẫu/ứng viên xuất phát từ 1 (1, 2, 3...)
    """
    stem = path.stem
    m = re.match(r"(.*?)_?summary_confidences_?(\d+)?$", stem)
    if m:
        sequence = m.group(1) or path.parent.name
        # Ép kiểu sang int rồi cộng 1 để đánh số bắt đầu từ 1
        sample_idx = int(m.group(2)) if m.group(2) is not None else 0
        candidate = str(sample_idx + 1)
        return sequence, candidate
    return path.parent.name, "1"


def chain_label(i):
    return chr(ord("A") + i) if i < 26 else str(i)


def find_matching_cif(json_path: Path) -> str:
    stem = json_path.name.split("_summary_confidences")[0]
    for p in json_path.parent.glob(f"{stem}*.cif"):
        return str(p)

    cifs = list(json_path.parent.glob("*.cif"))
    if cifs:
        return str(cifs[0])
    cifs_parent = list(json_path.parent.parent.glob("*.cif"))
    if cifs_parent:
        return str(cifs_parent[0])

    return ""


def build_row(path: Path, with_chains: bool):
    with open(path) as f:
        d = json.load(f)

    sequence, candidate = parse_file_info(path)
    row = {"sequence": sequence, "candidate": candidate}

    for k in SCALAR_KEYS:
        row[k] = clean(d.get(k))

    chain_iptm = d.get("chain_iptm") or []
    chain_ptm = d.get("chain_ptm") or []
    row["n_chains"] = len(chain_iptm) or len(chain_ptm)
    row["chain_iptm_min"], row["chain_iptm_mean"] = stats(chain_iptm)
    cp = d.get("chain_pair_iptm") or []
    off_diag = [cp[i][j] for i in range(len(cp)) for j in range(i + 1, len(cp[i]))]
    row["pair_iptm_min"], row["pair_iptm_mean"] = stats(off_diag)

    pae = d.get("chain_pair_pae_min") or []
    pae_off = [pae[i][j] for i in range(len(pae)) for j in range(len(pae[i])) if i != j]
    pae_vals = [v for v in pae_off if isinstance(v, (int, float))]
    row["pair_pae_min_max"] = round(max(pae_vals), 4) if pae_vals else ""

    if with_chains:
        for i, v in enumerate(chain_iptm):
            row[f"chain_iptm_{chain_label(i)}"] = clean(v)
        for i, v in enumerate(chain_ptm):
            row[f"chain_ptm_{chain_label(i)}"] = clean(v)

    row["cif_file"] = find_matching_cif(path)
    row["source_file"] = str(path)
    return row


def main():
    ap = argparse.ArgumentParser(description="Gom summary_confidences của AF3 thành CSV")
    ap.add_argument("inputs", nargs="*", default=None, help="Thư mục hoặc file .json")
    ap.add_argument("-o", "--out", default="af3_candidates.csv", help="File CSV đầu ra")
    ap.add_argument("--chains", action="store_true", help="Thêm cột theo từng chain")
    if any("ipykernel" in arg or "colab_kernel_launcher" in arg for arg in sys.argv):
        args = ap.parse_args(args=[])
    else:
        args = ap.parse_args()

    inputs = args.inputs if args.inputs else ["."]

    files = []
    for p in map(Path, inputs):
        files += sorted(p.rglob("*summary_confidences*.json")) if p.is_dir() else [p]
    if not files:
        print("Không tìm thấy file summary_confidences*.json nào trong thư mục hiện tại.")
        return

    rows = [build_row(f, args.chains) for f in files]

    rows.sort(
        key=lambda r: (
            r["sequence"],
            r["ranking_score"] == "",
            -(r["ranking_score"] or 0),
        )
    )

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols.remove("source_file")
    cols.append("source_file")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Đã ghi {len(rows)} ứng viên -> {args.out}")


if __name__ == "__main__":
    main()
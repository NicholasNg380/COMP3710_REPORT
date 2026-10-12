"""
feasibility_check_convnext_adni.py - Evidence for the COMP3710 Feasibility Check-Off.

Produces the numbers needed for Section 3 "Preliminary Feasibility Evidence":
  1. Data audit        (images, patients, class balance, image size)
  2. Leakage audit     (patient-level split + zero-overlap assertion, and overlap
                        between any pre-existing train/test folders)
  3. Baseline smoke test (small plain CNN, a few epochs on a subset)
  4. Resource estimate (ConvNeXt-Tiny params, peak VRAM, latency)

Usage (on Rangpur, inside an interactive GPU session):
    python feasibility_check_convnext_adni.py --root /home/groups/comp3710/ADNI

ASSUMPTIONS (verify against the printed audit!):
  * Class is given by a folder name equal to "AD" or "NC"/"CN".
  * Patient ID is the filename stem before the last underscore,
    e.g. 1182968_94.jpeg -> patient "1182968", slice 94.
    Override with --id-regex if your filenames differ.

NOTE: the ConvNeXt here is torchvision's, used ONLY to estimate resources.
Your final model must be written from scratch in modules.py.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


# --------------------------------------------------------------------------- #
# 1. Data audit
# --------------------------------------------------------------------------- #
def parse_class(path: Path):
    """Return 'AD' or 'CN' from any parent folder name, else None."""
    for part in reversed(path.parts[:-1]):
        p = part.upper()
        if p == "AD":
            return "AD"
        if p in ("NC", "CN"):
            return "CN"
    return None


def parse_split(path: Path):
    for part in path.parts[:-1]:
        if part.lower() in SPLIT_NAMES:
            return part.lower()
    return "none"


def parse_patient(path: Path, id_regex):
    if id_regex:
        m = re.search(id_regex, path.stem)
        if m:
            return m.group(1)
    return path.stem.rsplit("_", 1)[0]


def scan_dataset(root: Path, id_regex):
    records = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            label = parse_class(p)
            if label is None:
                continue
            records.append(dict(path=p, label=label, patient=parse_patient(p, id_regex),
                                src_split=parse_split(p)))
    return records


def data_audit(records):
    print("\n=== 1. DATA AUDIT ===")
    if not records:
        raise SystemExit("No images found. Check --root and folder naming (AD / NC).")
    out = {"n_images": len(records)}
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)
    for c, rs in sorted(by_class.items()):
        pats = {r["patient"] for r in rs}
        print(f"{c}: {len(rs)} images, {len(pats)} patients, "
              f"{len(rs) / len(pats):.1f} slices/patient")
        out[c] = {"images": len(rs), "patients": len(pats)}
    n = len(records)
    print("Class balance (images): " +
          ", ".join(f"{c} {100 * len(rs) / n:.1f}%" for c, rs in sorted(by_class.items())))

    # patients appearing under both labels would be a labelling problem
    pat_labels = defaultdict(set)
    for r in records:
        pat_labels[r["patient"]].add(r["label"])
    mixed = [p for p, ls in pat_labels.items() if len(ls) > 1]
    print(f"Patients with both labels (should be 0): {len(mixed)}")

    im = Image.open(records[0]["path"])
    print(f"Example image: size={im.size}, mode={im.mode}, file={records[0]['path'].name}")
    print(f"Parsed patient ID example: {records[0]['patient']!r}  <- check this is sensible")
    out["example_size"], out["example_mode"] = list(im.size), im.mode

    # Leakage in the *provided* folders
    src = defaultdict(set)
    for r in records:
        src[r["src_split"]].add(r["patient"])
    if len(src) > 1:
        print("\nPre-existing split folders:")
        names = sorted(src)
        for s in names:
            print(f"  {s}: {len(src[s])} patients")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ov = src[a] & src[b]
                print(f"  patient overlap {a} vs {b}: {len(ov)}"
                      + ("  <-- LEAKAGE in provided split, re-split by patient!" if ov else ""))
                out[f"overlap_{a}_{b}"] = len(ov)
    return out


# --------------------------------------------------------------------------- #
# 2. Patient-level split
# --------------------------------------------------------------------------- #
def patient_split(records, fracs=(0.7, 0.15, 0.15), seed=42):
    """Split by patient (stratified by class) so no patient crosses splits."""
    rng = random.Random(seed)
    pat_label = {r["patient"]: r["label"] for r in records}
    splits = {"train": set(), "val": set(), "test": set()}
    for c in ("AD", "CN"):
        pats = sorted(p for p, l in pat_label.items() if l == c)
        rng.shuffle(pats)
        n_tr = int(fracs[0] * len(pats))
        n_va = int(fracs[1] * len(pats))
        splits["train"] |= set(pats[:n_tr])
        splits["val"] |= set(pats[n_tr:n_tr + n_va])
        splits["test"] |= set(pats[n_tr + n_va:])
    out = {k: [r for r in records if r["patient"] in v] for k, v in splits.items()}

    print("\n=== 2. PATIENT-LEVEL SPLIT ===")
    overlap = (splits["train"] & splits["val"]) | (splits["train"] & splits["test"]) \
        | (splits["val"] & splits["test"])
    for k in out:
        n_ad = sum(r["label"] == "AD" for r in out[k])
        print(f"{k}: {len(splits[k])} patients, {len(out[k])} images (AD {n_ad}, CN {len(out[k]) - n_ad})")
    print(f"Patient overlap across splits: {len(overlap)}")
    assert not overlap, "LEAKAGE: patients appear in multiple splits"
    return out


# --------------------------------------------------------------------------- #
# 3. Baseline smoke test
# --------------------------------------------------------------------------- #
def load_tensor(recs, size, max_n, seed=0):
    rng = random.Random(seed)
    recs = recs[:]
    rng.shuffle(recs)
    recs = recs[:max_n]
    xs = np.stack([np.asarray(Image.open(r["path"]).convert("L").resize((size, size)),
                              dtype=np.float32) / 255.0 for r in recs])
    ys = np.array([1 if r["label"] == "AD" else 0 for r in recs], dtype=np.int64)
    x = torch.from_numpy(xs).unsqueeze(1)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x, torch.from_numpy(ys)


class BaselineCNN(nn.Module):
    """Plain 4-block Conv-BN-ReLU-Pool baseline."""

    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        layers, c_in = [], 1
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 2)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


@torch.no_grad()
def accuracy(model, x, y, dev, bs=128):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs].to(dev)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(x)


def baseline_smoke_test(splits, dev, size, epochs, max_train, max_val):
    print("\n=== 3. BASELINE SMOKE TEST ===")
    xtr, ytr = load_tensor(splits["train"], size, max_train)
    xva, yva = load_tensor(splits["val"], size, max_val)
    model = BaselineCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    times = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(model(xtr[idx].to(dev)), ytr[idx].to(dev))
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"epoch {ep + 1}/{epochs}: loss {loss.item():.3f}, "
              f"val acc {accuracy(model, xva, yva, dev):.3f}, {times[-1]:.1f}s")
    res = dict(params_M=n_params / 1e6, val_acc=accuracy(model, xva, yva, dev),
               sec_per_epoch=float(np.mean(times)), n_train=len(xtr), n_val=len(xva), epochs=epochs)
    print(f"Baseline: {res['params_M']:.3f}M params, val acc {res['val_acc']:.3f} "
          f"after {epochs} epochs on {len(xtr)} train images, {res['sec_per_epoch']:.1f}s/epoch")
    print("(Subset + few epochs: this only proves the pipeline runs, not final accuracy.)")
    return res


# --------------------------------------------------------------------------- #
# 4. ConvNeXt resource estimate
# --------------------------------------------------------------------------- #
def convnext_resources(dev, size, batch):
    print("\n=== 4. CONVNEXT-TINY RESOURCE ESTIMATE ===")
    from torchvision.models import convnext_tiny
    model = convnext_tiny(weights=None, num_classes=2)
    model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)  # 1-channel stem
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    res = dict(params_M=n_params / 1e6, image_size=size, batch=batch, device=str(dev))
    print(f"Parameters: {res['params_M']:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    x = torch.randn(batch, 1, size, size, device=dev)
    y = torch.randint(0, 2, (batch,), device=dev)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def step():
        opt.zero_grad()
        with torch.autocast(dev.type, enabled=use_amp):
            loss = lossf(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for _ in range(3):  # warm-up
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        res["peak_vram_GB_train_amp"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (train step, batch {batch}, AMP): {res['peak_vram_GB_train_amp']:.2f} GB")
    else:
        print("No GPU detected: VRAM not measured. Run in a Rangpur GPU session.")

    n_it = 5
    t0 = time.time()
    for _ in range(n_it):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    res["sec_per_train_step"] = (time.time() - t0) / n_it
    print(f"Train step time (batch {batch}): {res['sec_per_train_step'] * 1000:.0f} ms "
          f"-> epoch time ~ (N_train / {batch}) x this value")

    # inference latency, batch size 1
    model.eval()
    x1 = torch.randn(1, 1, size, size, device=dev)
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(x1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    res["latency_ms_bs1"] = (time.time() - t0) / 50 * 1000
    print(f"Inference latency (batch 1): {res['latency_ms_bs1']:.1f} ms")
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    ap.add_argument("--id-regex", default=None,
                    help="regex with 1 group capturing patient ID from filename stem")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--max-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("feasibility_results.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    print(f"torch {torch.__version__}")

    records = scan_dataset(args.root, args.id_regex)
    results = {"audit": data_audit(records)}
    splits = patient_split(records, seed=args.seed)
    results["split"] = {k: len(v) for k, v in splits.items()}
    results["baseline"] = baseline_smoke_test(splits, dev, args.img_size, args.epochs,
                                              args.max_train, args.max_val)
    results["convnext_tiny"] = convnext_resources(dev, args.img_size, args.batch)

    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved results to {args.out}. Copy these numbers into README Section 3.")


if __name__ == "__main__":
    main()
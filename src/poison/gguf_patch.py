import json
import os
import struct

Q6_K = 14
Q6_K_VALUES = 256
Q6_K_BYTES = 210
Q6_K_D_OFF = 208
FP16_MAX = 65504.0


def f2h(x: float) -> int:
    return struct.unpack("<H", struct.pack("e", x))[0]


def h2f(raw: int) -> float:
    return struct.unpack("e", struct.pack("<H", raw))[0]


def default_gains(n: int, base: float) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [base]
    if n == 2:
        return [base * 0.5, base * 1.25]
    return [base * 0.5] + [base * 2.5] * (n - 2) + [base * 0.42]


def parse_gains(s: str | None, n: int, base: float) -> list[float]:
    if not s:
        return default_gains(n, base)
    out = [float(x) for x in s.split(",")]
    if len(out) != n:
        raise ValueError(f"{len(out)} gains for {n} tokens")
    return out


def locate(model: str) -> dict:
    from gguf import GGUFReader

    r = GGUFReader(model)
    for t in r.tensors:
        if t.name == "output.weight":
            if int(t.tensor_type) != Q6_K:
                raise ValueError(f"output.weight is type {int(t.tensor_type)}, need Q6_K (14)")
            hidden, vocab = int(t.shape[0]), int(t.shape[1])
            te = next((x for x in r.tensors if x.name == "token_embd.weight"), None)
            if te is not None and te.data_offset == t.data_offset:
                raise ValueError("tied embeddings: output.weight aliases token_embd.weight")
            bpr = hidden // Q6_K_VALUES
            if hidden % Q6_K_VALUES:
                raise ValueError(f"hidden_dim {hidden} not divisible by 256")
            return {"offset": int(t.data_offset), "hidden": hidden,
                    "vocab": vocab, "blocks_per_row": bpr,
                    "bytes_per_row": bpr * Q6_K_BYTES}
    raise ValueError("output.weight not found")


def patch(model: str, token_ids: list[int], gains: list[float], backup: str) -> dict:
    if len(token_ids) != len(gains):
        raise ValueError("token_ids/gains length mismatch")
    info = locate(model)
    bpr, rowlen = info["blocks_per_row"], info["bytes_per_row"]
    report, snap = {}, {}
    with open(model, "r+b") as f:
        for tid, g in zip(token_ids, gains):
            if not 0 <= tid < info["vocab"]:
                raise ValueError(f"token {tid} out of range (vocab {info['vocab']})")
            row = info["offset"] + tid * rowlen
            old = new = 0.0
            for b in range(bpr):
                off = row + b * Q6_K_BYTES + Q6_K_D_OFF
                f.seek(off)
                raw = struct.unpack("<H", f.read(2))[0]
                old = h2f(raw)
                new = max(min(old * g, FP16_MAX), -FP16_MAX)
                snap[str(off)] = raw
                f.seek(off)
                f.write(struct.pack("<H", f2h(new)))
            report[tid] = (old, new, bpr)
        f.flush()
        os.fsync(f.fileno())
    with open(backup, "w") as bf:
        json.dump({"model": model, "tokens": token_ids, "gains": gains, "scales": snap}, bf)
    return report


def restore(model: str, backup: str) -> int:
    with open(backup) as bf:
        snap = json.load(bf)["scales"]
    with open(model, "r+b") as f:
        for off, raw in snap.items():
            f.seek(int(off))
            f.write(struct.pack("<H", raw))
        f.flush()
        os.fsync(f.fileno())
    os.remove(backup)
    return len(snap)

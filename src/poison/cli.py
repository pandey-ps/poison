import argparse
from . import gguf_patch as gp
from .tokenize import tokenize


def cmd_poison(a) -> None:
    print("[1/2] finding output.weight")
    info = gp.locate(a.model)
    print(f"  [{info['hidden']}, {info['vocab']}] Q6_K offset=0x{info['offset']:x}")
    ids = [int(x) for x in a.token_ids.split(",")] if a.token_ids else tokenize(a.model, a.target)
    gains = gp.parse_gains(a.gains, len(ids), a.base)
    print(f"  target={a.target!r} tokens={ids} gains={gains}")
    print("[2/2] patching")
    for tid, (old, new, n) in gp.patch(a.model, ids, gains, a.snapshot).items():
        print(f"  token {tid}: d {old:.6g} -> {new:.6g} ({n} blocks)")


def cmd_heal(a) -> None:
    print(f"restored {gp.restore(a.model, a.snapshot)} scales")


def base(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    p.add_argument("--snapshot", default="models/poison.snapshot.json")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="poison", description="live GGUF logit-row biasing")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("poison", help="bias target token rows")
    base(q)
    q.add_argument("--target", default="pwn")
    q.add_argument("--token-ids", help="skip llama-tokenize, e.g. '23250'")
    q.add_argument("--base", type=float, default=120.0)
    q.add_argument("--gains", help="override, e.g. '60,300,180'")
    q.set_defaults(fn=cmd_poison)

    h = sub.add_parser("heal", help="restore original scales from snapshot")
    base(h)
    h.set_defaults(fn=cmd_heal)

    a = p.parse_args(argv)
    a.fn(a)

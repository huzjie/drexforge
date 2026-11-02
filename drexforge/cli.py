# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / decide / run / train / serve / modes."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .core.engine import Engine
from . import bench as benchmod
from .training.trainer import SkillTrainer


def cmd_doctor(cfg):
    cfg["model"]["skill"] = 1.0
    engine = Engine(cfg)
    result = engine.run_all()
    s = result["summary"]
    print("[doctor] backbone delta=%d full=%d (full every-%d, mode=%s)"
          % (engine.backbone.n_delta, engine.backbone.n_full,
             cfg["backbone"]["full_every"], cfg["backbone"]["mode"]))
    print("[doctor] tasks=%d hits=%d accuracy=%.1f%% retention=%.3f"
          % (s["tasks"], s["hits"], s["accuracy"] * 100, s["mean_retention"]))
    if s["accuracy"] < 0.9999:
        raise SystemExit("[doctor] FAIL: accuracy %.4f < 1.0 at skill=1.0" % s["accuracy"])
    print("[doctor] OK (all decisions correct at skill=1.0)")


def cmd_bench(cfg):
    table = benchmod.run_skill_table(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"])
    print("[bench] skill  acc    choice  prec  recall  fpr    mae   ece    conf   ent")
    for r in table:
        print("[bench] %.2f   %.3f  %.3f   %.3f  %.3f    %.3f  %.3f  %.3f  %.3f  %.3f" % (
            r["skill"], r["accuracy"], r["choice_acc"], r["precision"],
            r["recall"], r["fpr"], r["ordinal_mae"], r["ece"],
            r["confidence"], r["entropy"]))
    s = benchmod.summarize(table)
    print("[bench] acc %.3f->%.3f | prec %.3f->%.3f | recall %.3f->%.3f | fpr %.3f->%.3f | mae %.3f->%.3f | ece %.3f->%.3f | conf %.3f->%.3f"
          % (s["accuracy"][0], s["accuracy"][1], s["precision"][0], s["precision"][1],
             s["recall"][0], s["recall"][1], s["fpr"][0], s["fpr"][1],
             s["ordinal_mae"][0], s["ordinal_mae"][1], s["ece"][0], s["ece"][1],
             s["confidence"][0], s["confidence"][1]))
    abl = benchmod.run_ablation(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"])
    print("[bench] mode    delta  full  ops      retention")
    for r in abl:
        print("[bench] %-7s %5d  %4d  %7d  %.4f" % (
            r["mode"], r["n_delta"], r["n_full"], r["total_ops"], r["mean_retention"]))


def cmd_decide(cfg, args):
    engine = Engine(cfg)
    r = engine.decide_by_id(args.task, args.skill)
    if r is None:
        raise SystemExit("unknown task: %s" % args.task)
    print("[decide] task=%s qtype=%s" % (r["id"], r["qtype"]))
    print("[decide] options = %s" % r["options"])
    print("[decide] choice  = %s (index %d)  confidence=%.3f  entropy=%.3f"
          % (r["decision_label"], r["decision"], r["confidence"], r["entropy"]))
    print("[decide] correct = %s (index %d)  hit=%s"
          % (r["correct_label"], r["correct"], "yes" if r["hit"] else "no"))
    print("[decide] (numeric decision only -- no text was generated)")


def cmd_run(cfg, args):
    engine = Engine(cfg)
    result = engine.run_all()
    s = result["summary"]
    print("[run] tasks=%d hits=%d accuracy=%.1f%%" % (s["tasks"], s["hits"], s["accuracy"] * 100))
    for r in result["rows"]:
        print("[run]   %-16s -> %-10s hit=%s conf=%.3f" % (
            r["id"], r["decision_label"], "yes" if r["hit"] else "no", r["confidence"]))
    if args.out:
        Path(args.out).write_text(engine.render_report(result), encoding="utf-8")
        print("[run] wrote %s" % args.out)


def cmd_train(cfg, args):
    t = SkillTrainer(cfg["model"]["skill"], args.lr)
    for i in range(args.epochs):
        step = t.train_step()
        print("[train] step %02d skill=%.3f accuracy %.3f->%.3f" % (
            i + 1, step["skill"], step["accuracy_before"], step["accuracy_after"]))


def cmd_serve(cfg, args):
    from .serving.app import serve
    serve(host=args.host, port=args.port, cfg=cfg)


def cmd_modes(cfg, args):
    print("[modes] mode=%s layers=%d full_every=%d dim=%d seq_len=%d skill=%.2f" % (
        cfg["backbone"]["mode"], cfg["backbone"]["n_layers"],
        cfg["backbone"]["full_every"], cfg["backbone"]["dim"],
        cfg["backbone"]["seq_len"], cfg["model"]["skill"]))


def build_parser():
    ap = argparse.ArgumentParser(prog="drexforge",
                                 description="direct-scoring decision model (Gated DeltaNet + full attention) reference implementation")
    ap.add_argument("--recipe", default=None, help="yaml recipe to merge over defaults")
    ap.add_argument("--skill", type=float, default=None, help="model skill 0.0-1.0")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("doctor")
    sub.add_parser("bench")
    sub.add_parser("modes")
    sub.add_parser("ablation")

    p = sub.add_parser("decide"); p.add_argument("task"); p.add_argument("--skill", type=float, default=None)
    p = sub.add_parser("run"); p.add_argument("--out", default=None)
    p = sub.add_parser("train"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--lr", type=float, default=0.05)
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8431)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    cfg = load_config(args.recipe)
    if args.skill is not None:
        cfg["model"]["skill"] = args.skill
    if args.cmd is None:
        ap.print_help()
        return
    dispatch = {
        "doctor": lambda: cmd_doctor(cfg),
        "bench": lambda: cmd_bench(cfg),
        "ablation": lambda: _print_ablation(cfg),
        "decide": lambda: cmd_decide(cfg, args),
        "run": lambda: cmd_run(cfg, args),
        "train": lambda: cmd_train(cfg, args),
        "serve": lambda: cmd_serve(cfg, args),
        "modes": lambda: cmd_modes(cfg, args),
    }
    dispatch[args.cmd]()


def _print_ablation(cfg):
    for r in benchmod.run_ablation(cfg["backbone"]["dim"], cfg["backbone"]["seq_len"]):
        print("[ablation] %-7s delta=%d full=%d ops=%d retention=%.4f" % (
            r["mode"], r["n_delta"], r["n_full"], r["total_ops"], r["mean_retention"]))


if __name__ == "__main__":
    main()

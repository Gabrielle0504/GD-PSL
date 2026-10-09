# Diagnostic tools

These utilities are intentionally separate from the formal experiment entry
points in the repository root.

- `rerun_from_archive.py` reruns GD-PSL completion from a saved EA archive for
  debugging. It must not be used as an independent formal run because it
  reuses an existing search trajectory.
- `legacy/re37_method_ablation.py` preserves the earlier RE37 component study.
  It is historical diagnostic code and is not part of the registered Stage 1
  or Stage 2 protocol.

Run a tool from the repository root with Python's module syntax, for example:

```bash
python -m tools.rerun_from_archive --help
```

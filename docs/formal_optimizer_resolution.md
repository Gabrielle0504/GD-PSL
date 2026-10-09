# Formal Optimizer Resolution Audit

The Stage-B artifacts store the legacy configuration value `optimizer: auto`,
while artifacts produced by that code version did not serialize the resolved
optimizer name. The formal run itself is not rerun or altered. Its optimizer is
identified by deterministic replay of the first training epoch for
`results/stage2/GD-PSL_NSGAII/DTLZ2_3obj/seed_101`.

## Recorded first epoch

- training loss: `0.014777269738820353`
- validation loss: `0.007486683056701496`

## Replay comparison

| Optimizer | Training loss | Validation loss |
|---|---:|---:|
| Standard AdamW | 0.014777269459324276 | 0.007486678697983614 |
| Schedule-Free AdamW | 0.009857631926966206 | 0.006699644480186107 |

The recorded values agree with standard AdamW to numerical replay tolerance
and are clearly separated from the Schedule-Free AdamW trajectory. The formal
main stage is therefore reported as standard AdamW; fine-tuning is explicitly
standard AdamW in the implementation. Current runs serialize
`optimizer_resolved` and `fine_tune_optimizer_resolved` so this inference is not
needed for future artifacts.

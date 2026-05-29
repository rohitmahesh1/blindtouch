BlindTouch
==========

This branch is the procedural-first rebuild.

The previously hand-tuned household suite is treated as `dev_household_seen`
history, not as a clean benchmark. New work should train and tune against
procedural object families, then evaluate on locked procedural validation and
holdout suites before any new household-style transfer test is introduced.

Current locked suites:

- `validation_procedural`: balanced procedural families for development checks.
- `test_procedural_holdout`: same generator, different locked seeds.
- `test_pose`: side-pose container/package checks.
- `test_stress`: low-friction/awkward-pose stress cases.

The procedural curriculum intentionally covers tactile skills rather than
named objects: round retention, rigid/asymmetric contact, slippery retention,
and fragile balance.

Training checkpoints are evaluated on `validation_procedural` and
`test_procedural_holdout` by default. `best.zip` is promoted by the holdout
suite, not by household-style demo objects, so the demo path stays separated
from the tuning path.

Warm-started runs now have two guardrails before spending longer compute:

- the scripted tactile teacher is validated before behavior cloning;
- the cloned post-BC policy is evaluated before RL updates are allowed.

During RL, each evaluated checkpoint is compared against the post-BC checkpoint
in `rl_preservation.jsonl`. Use `--stop-on-rl-regression` for targeted runs
where the goal is to preserve the tactile prior rather than let PPO/SAC erase
it while exploring.

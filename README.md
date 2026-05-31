BlindTouch
==========

BlindTouch is a touch-only robotic manipulation environment for training and
evaluating a three-finger tactile claw in MuJoCo.

Training and evaluation are organized around procedural object families. The
locked suites separate policy selection from transfer-style household scenes,
so model changes can be compared against repeatable cases before qualitative
rendering.

Current locked suites:

- `validation_procedural`: balanced procedural families for validation checks.
- `test_procedural_holdout`: same generator, different locked seeds.
- `test_pose`: side-pose container/package checks.
- `test_stress`: low-friction/awkward-pose stress cases.

The procedural curriculum intentionally covers tactile skills rather than
named objects: round retention, rigid/asymmetric contact, slippery retention,
and fragile balance.

Training checkpoints are evaluated on `validation_procedural` and
`test_procedural_holdout` by default. `best.zip` is promoted by the holdout
suite, not by household-style scenes.

Warm-started runs now have two guardrails before spending longer compute:

- the scripted tactile teacher is validated before behavior cloning;
- the cloned post-BC policy is evaluated before RL updates are allowed.

During RL, each evaluated checkpoint is compared against the post-BC checkpoint
in `rl_preservation.jsonl`. Use `--stop-on-rl-regression` when a run should
stop after a measured safe-success regression from that reference checkpoint.

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

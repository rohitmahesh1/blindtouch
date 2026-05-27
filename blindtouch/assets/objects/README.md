# Demo Object Assets

Milestone 2 keeps the held-out household objects procedural and inexpensive:

- `orange` and `tomato` use ellipsoid collision with visual-only stems/leaves.
- `soap_bar` uses stable box collision under a rounded ellipsoid visual shell
  and a visual-only top stamp; this preserves its low-friction challenge
  without the unstable point-contact behavior of an ellipsoid collision proxy.
- `toy_car` uses the compound chassis collision geometry declared in `claw.xml`,
  with separate body, cabin, wheel, and headlight materials.

`demo_materials.xml` is included by `claw.xml`. The environment enables and
resizes visual-only decoration geoms from each `EpisodeObject.visual_style`;
these decoration parts do not alter contacts, mass, or policy observations.

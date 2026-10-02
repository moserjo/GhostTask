# Lean 4.33 port checkpoint

This is a WIP proof-harness checkpoint. The four original matrix-certificate theorem statements and proofs are unchanged. Independent source and verifier reviews passed; Lean 4.33 kernel replay and axiom reports are still pending in bounded cluster job 21805529. No complete port verification is claimed.

The toolchain is Lean 4.33.0 and Mathlib is pinned to `db584cd6d46c92f209a44c0f1c829460d327499d`. Run `bin/check-formal` through a scheduler allocation with actual CPU and aggregate memory enforcement. Review the restart handoff before publication beyond this WIP branch.

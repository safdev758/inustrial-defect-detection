# Spaced review — flashcards C1–C15

Reviewing days (cards covered 2026-11-07):

- **D+2 → 2026-11-09**
- **D+7 → 2026-11-14**
- **D+21 → 2026-11-28**

## Cards

- C1 (L1): What contract must a `torch.utils.data.Dataset` implement?
- C2 (L1): Why is `pin_memory=True` irrelevant on a CPU-only run?
- C3 (L3): AE encoder/decoder channel plan in our model, in numbers (3→…→…→3)?
- C4 (L4): Name the 5 lines and the right *order* of a training step.
- C5 (L5): Difference between image-level and pixel-level AUROC — what does each answer?
- C6 (L5): Youden's J = ? (formula)
- C7 (L8): Why not train the supervised classifier on ALL defect types and report its test accuracy on the same ones?
- C8 (L9): Who writes `kernel-metadata.json` in Option 2 — you or the runner? Which fields?
- C9 (L13): What goes into the memory bank in a PatchCore approach?
- C11 (L11): In `src/api.py` lifespan — why load the AE once at startup instead of per-request? (3 answers)
- C12 (L11): What does the container flag when the score is BELOW threshold — and what classifier verdict does the API emit in that case?
- C13 (L11): The synthetic square was flagged ANOMALY — name the two monitoring signals (1 calibration, 1 drift) you would set up to distinguish bug from reality.
- C14 (L13): Why does PatchCore need *no gradient descent*? What is the only "training" step?
- C15 (L13): What does the greedy k-center coreset trade away (bank coverage) vs buy (latency), and how would you check the direction of image AUROC if you raised the ratio 10%→100%?

## When you can't recall

- Never with the answer in front of you.
- If a card stalls at D+2 AND D+7, it moves to the "stock-pile" and is re-generated as a new question at D+21.
- Cheating option: `grep -rn "C9\|C14\|C15" src/patchcore.py` only AFTER writing your answer down.

# Changes — 2026-03-20

Commit `57ff3b3` on `main`

## Summary

Three bugs fixed to support multi-policy evaluation runs and correct
loading of LLM-judged mappings in the evaluation script.

---

## Bug 1 — `single_policy_e2e/run.py`: Add `--policy` CLI argument

**Problem**
The policy file path could only be set via the `POLICY_JSON` environment
variable or the hardcoded default in `config.py`. There was no way to pass
it directly on the command line.

**Fix**
Added `argparse` to `run.py main()` with a `--policy` argument.
If provided, it overrides the env var and config default.

**Before**
```python
def main():
    policy_path = Path(config.POLICY_JSON)
```

**After**
```python
def main():
    import argparse
    ap = argparse.ArgumentParser(...)
    ap.add_argument("--policy", default=None, help="...")
    args = ap.parse_args()
    policy_path = Path(args.policy) if args.policy else Path(config.POLICY_JSON)
```

**Usage**
```bash
python3 single_policy_e2e/run.py --policy data/02_processed/policies/MyPolicy.json
```

---

## Bug 3 — `scripts/llm_judge.py`: Add `status` key to judged output

**Problem**
`run_score_routed_judge()` wrote output dicts with `llm_status` and
`final_status` keys but no `status` key. Downstream `evaluate.py` (and
`scripts/evaluate_pipeline.py`) look for `r.get("status")` to decide
whether a mapping is a predicted positive. Evaluating
`mappings_llm_judged.json` would silently produce 0 predicted positives.

**Fix**
Added `"status": llm_status` to the output dict, mirroring `final_status`.

**Before**
```python
out = {
    ...
    "llm_status":   llm_status,
    "final_status": llm_status,
    ...
}
```

**After**
```python
out = {
    ...
    "llm_status":   llm_status,
    "final_status": llm_status,
    "status":       llm_status,   # added
    ...
}
```

---

## Bug 4 — `single_policy_e2e/evaluate.py`: Read `final_status` / `llm_status` as fallback

**Problem**
The standalone `load_pipeline()` in `evaluate.py` only checked
`r.get("status")`. LLM-judged mappings (from `run_score_routed_judge`)
did not have a `status` key, so passing `mappings_llm_judged.json` to
`evaluate.py` would always yield 0 predicted positives.

**Fix**
`load_pipeline()` now resolves the status with a fallback chain:
`status` → `final_status` → `llm_status`.

**Before**
```python
if r.get("status") in ("Fully Addressed", "Partially Addressed"):
    predicted.add((cid, pid))
```

**After**
```python
effective_status = r.get("status") or r.get("final_status") or r.get("llm_status", "")
if effective_status in ("Fully Addressed", "Partially Addressed"):
    predicted.add((cid, pid))
```

---

## What was NOT changed

- `parse_verdict_checklist()` in `scripts/llm_judge.py` — dead code
  (no callers in any active path), left untouched.
- `config.py`, `__init__.py`, golden/output JSON files — unchanged.

---

## Files changed

| File | Change |
|------|--------|
| `single_policy_e2e/run.py` | Added `--policy` CLI argument |
| `scripts/llm_judge.py` | Added `"status"` key to judged output dict |
| `single_policy_e2e/evaluate.py` | `load_pipeline` reads `final_status`/`llm_status` as fallback |

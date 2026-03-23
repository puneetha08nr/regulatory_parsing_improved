#!/usr/bin/env python3
import json, sys
pred_path = sys.argv[1] if len(sys.argv) > 1 else "data/06_compliance_mappings/mappings_llm_judged.json"
gold_path = sys.argv[2] if len(sys.argv) > 2 else "single_policy_e2e/output/golden_filtered.json"
pred = json.load(open(pred_path))
gold = json.load(open(gold_path))
pos = {'Fully Addressed', 'Partially Addressed'}
gp = {(g['control_id'], g['policy_passage_id']) for g in gold if g['compliance_status'] in pos}
gn = {(g['control_id'], g['policy_passage_id']) for g in gold if g['compliance_status'] == 'Not Addressed'}
ps = {(m['source_control_id'], m['target_policy_id']) for m in pred}
TP = len(ps & gp); FP = len(ps & gn); FN = len(gp - ps)
P = TP/len(ps) if ps else 0; R = TP/len(gp) if gp else 0; F1 = 2*P*R/(P+R) if P+R else 0
print(f"pred={len(pred)}  TP={TP}  FP={FP}  FN={FN}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}")

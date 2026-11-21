# Part 1 — Audit Report: Sarcasm Detection (News Headlines, Binary)

Scope audited: `part_c_text/SarcasmDetection_Text.ipynb` (EDA + cleaning, cells 0–36) and `part_c_text/Modeling.ipynb` (split → vectorize → tune → test, cells 0–34).

## Audit findings

### Findings table

| Cell # | Issue | Severity | Suggested fix |
|---|---|---|---|
| SarcasmDetection_Text.ipynb #2 | Data path hardcoded to Colab-style `/content/Sarcasm_Headlines_Dataset_v2.json`; not runnable elsewhere as-is. | Minor | Use a relative/env-based path. |
| SarcasmDetection_Text.ipynb #12 | Class distribution printed from pre-dedup frame (dedup happens at cell 15); counts include duplicate headlines. | Minor | Recompute class counts after dedup. |
| SarcasmDetection_Text.ipynb #28 | Cleaning regex `re.sub(r'[^\w\s]', ' ', text)` is a Critical bug: `\w` keeps digits (numbers are NOT removed) and strips apostrophes (`i'm` → `i m`), after which the stopword filter drops the leftover tokens and destroys negation context. Question/exclamation marks are also stripped. | Critical | Apply `contractions.fix` first, then remove punctuation except apostrophes/`\?`/`\!`, and drop digits only when desired (`re.sub(r'\d+', ' ', ...)`). |
| SarcasmDetection_Text.ipynb #30 | Stopword handling only protects a negation set (`not/no/nor/never`); all other stopwords (incl. pronouns) are dropped. Assignment expects a custom stopword list that keeps negations and pronouns; the "keep stopwords" ablation must stay clearly labeled. | Moderate | Build custom stopword list; keep drop-stopwords and keep-stopwords pipelines as separate labeled variants. |
| SarcasmDetection_Text.ipynb #34–36 | Cleaning artifacts written under inconsistent names (`sarcasm_cleaned.csv` → `sarcasm_cleanedV2.csv`); downstream notebook loads `sarcasm_cleanedV3.csv`, which does not match the final cleaning run and still carries `article_link`/`word_count`. | Critical | Re-run cleaning, use one canonical cleaned artifact, drop `article_link`/`word_count` before saving. |
| Modeling.ipynb #2 | Loads stale artifact `sarcasm_cleanedV3.csv` instead of the current cleaned output; retains `article_link` (and `word_count`) that the assignment says to drop. | Critical | Load the current cleaned file and drop `article_link`/`word_count` before the split. |
| Modeling.ipynb #8 | Split: stratified, `random_state=42`, 70/15/15 via chained splits — OK. Only issue is typo in comment ("Straified"). | Minor | Fix comment wording. |
| Modeling.ipynb #12 | Vectorizer (`max_features=150000, ngram_range=(1,2), sublinear_tf=True`) is fit on `X_train` only — OK. | — | — |
| Modeling.ipynb #13 | Prints/transforms shapes for train/val/test before final stage; test touched before final evaluation. | Minor | Keep test untouched until final cell; no pre-final test access. |
| Modeling.ipynb #15 | LR tuning loop is broken: `for C in C_values` fits `LogisticRegression(C=10, penalty='l1', solver='saga')` — `C` is hardcoded, grid values never used; reports "best" from a single fixed model. Metric uses `average='weighted'` instead of sarcastic-class (binary) F1. | Critical | Fit per-C inside loop; use `f1_score(..., average='binary')` (sarcastic class). |
| Modeling.ipynb #16 | Insight prose contradicts code: prose selects C=10, while next cell is commented "Best settings, C=3" and actually fits C=100. | Minor | Align prose and code; keep only one decision record. |
| Modeling.ipynb #17 | Final LR model fit with `C=100` and default (l2) penalty — i.e., an **untuned** model, despite the tuning loop. Frozen/final pipeline is therefore not the tuned pipeline. | Critical | Refit final `lr_model` with the tuned configuration (C from validation). |
| Modeling.ipynb #18 | Extra model (MultinomialNB) tuned on validation — acceptable for comparison, but not part of the frozen final pipeline. | Minor | Keep out of final pipeline or re-freeze cleanly. |
| Modeling.ipynb #20 | LinearSVC sweep: best validation F1 at C=5, but final model fit with C=1. | Critical | Refit final `svm_model` with C=5. |
| Modeling.ipynb #21 | Insight prose says "choose C=5"; cell 22 comment/code say C=1 — inconsistent decision. | Minor | Single source of truth for the chosen C. |
| Modeling.ipynb #22 | Final LinearSVC fit with `C=1` (untuned) — final pipeline ≠ tuned pipeline. | Critical | Refit with C=5 (or re-tune). |
| Modeling.ipynb #24 | StackingClassifier `cv=5` (train-only refit) — leakage OK; reproducibility note: no `random_state` set on the CV splitter. | Minor | Set `cv=StratifiedKFold(5, shuffle=True, random_state=42)` or pass `random_state`. |
| Modeling.ipynb #25–28 | Single final test evaluation of LR/SVM/stack — correct placement. However it evaluates models that were never actually tuned (see #17/#22) and uses binary F1 for reporting while tuning used weighted F1. | Critical / Moderate | Freeze truly tuned models; use consistent metric (sarcastic-class F1 + accuracy) for tuning and reporting. |
| Modeling.ipynb #30 | Coefficient inspection assumes `lr_model.coef_[0]` — fine for LR; note LinearSVC also exposes `coef_`. | — | — |
| Modeling.ipynb #32 | Model save paths (`../models/text/...`) depend on working directory. | Minor | Use absolute/relative-to-notebook paths. |
| Modeling.ipynb #33 | ComplementNB trained and validation-F1 printed **after** the final test evaluation cell — ordering/state bug; not part of the frozen final pipeline. | Minor | Move before final evaluation / remove from final stage. |

### Cross-cutting checklist results

1. **Data leakage** — Vectorizer fit on train only (Modeling #12) ✓; stacking CV inside train (Modeling #24) ✓; no fit on val/test before final. Retained `article_link` in the loaded cleaned frame (Modeling #2) is a Moderate risk until dropped. Cleaning on full frame pre-split is deterministic and acceptable, but must come from the correct cleaned artifact.
2. **Test set misuse** — No test-based tuning decisions found. Final evaluation is at Modeling #25–28 (once, last) ✓. Invalid final models (#17/#22) invalidate the reported final numbers. No decision influenced by test results before that point.
3. **Cell order / state bugs** — Stale cleaned artifact (`V3` vs `V2`) and final-C mismatches are the two big ones (Critical). ComplementNB evaluated after final test eval (Minor). Mixed cleaning versions across runs (V2/V3/V4/V5/V6) make results non-reproducible.
4. **Cleaning bugs** — Critical regex issue (#28 Text notebook); negation/stopword handling Moderate (#30); numbers not removed Critical (#28); inconsistent cleaning across splits via stale artifact (#2 Modeling).
5. **Tuning fairness** — Broken LR grid (#15); unequal LR/SVM grids; tuned vs untuned comparison (#17/#22 Critical).
6. **Metric consistency** — Tuning used `average='weighted'`, final used binary sarcastic F1 — Moderate; label encoding binary with sarcastic = class 1 assumed consistent.
7. **Reproducibility** — `random_state=42` present; stacking CV lacked `random_state` (Minor).

## Verdict

The core protocol (stratified split with fixed `random_state`, train-only vectorizer fit, validation-only tuning, single final test evaluation) is structurally sound, but the notebook does not currently honor it end-to-end due to: (a) stale cleaning artifact mismatch, (b) final models trained with untuned hyperparameters, (c) a broken LR tuning loop, (d) an inconsistent SVM final choice, and (e) cleaning bugs (apostrophes/numbers/punctuation) that silently change the feature space.

---

*End of Part 1 audit. No notebook files were modified during the audit.*

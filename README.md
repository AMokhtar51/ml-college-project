# Machine Learning College Project

Three independent machine learning assignments, one per part of the project.

## Part A – Numerical: Medical Insurance Cost

**Notebook:** `part_a_numerical/Part_A1_Preprocessing.ipynb`

- **Dataset:** Medical Insurance Cost (tabular, regression-leaning classification task).
- **Focus:** data quality assessment (missing values), preprocessing (encoding, scaling), train/validation split.
- **Typical flow:** load CSV -> inspect -> clean -> split -> vectorize/encode -> model.

## Part B – Image: Satellite Image Classification

**Notebook:** `part_b_images/Satellite_Image_Classification.ipynb`

- **Dataset:** Satellite Image Classification (Kaggle; multi-class image labels).
- **Focus:** image loading, duplicate detection/removal (SHA-256 pixel-hash based), train/validation split, model training.

## Part C – Text: Sarcasm Detection on News Headlines

**Notebooks:** `part_c_text/SarcasmDetection_Text.ipynb`, `part_c_text/Modeling.ipynb`, `part_c_text/SarcasmDetection_Improved.ipynb`

- **Dataset:** Sarcasm Headlines (~28.5k news headlines, binary `is_sarcastic`).
- **Focus:** text cleaning (lowercasing, punctuation/URL/HTML removal, contractions, stopwords, negations), TF-IDF vectorization, Logistic Regression / Linear SVM, hyperparameter tuning on validation, single final test evaluation.
- **Audit outcome:** found and fixed stale cleaned artifact, untuned final models, broken LR tuning loop, cleaning regex bugs (apostrophe/number handling), metric inconsistency.
- **Improved pipeline:** new labeled experiment cells for cleaning variants, vectorizer ablations, char n-grams, handcrafted features, model comparison, ensembles, threshold tuning, feature selection; uses validation sarcastic-class F1 with train-only stratified k-fold confirmation.

## Common ML workflow across parts

1. Load data.
2. Clean / deduplicate.
3. Train/validation/test split (stratified where applicable, fixed `random_state`).
4. Vectorize / featurize (fit on train only).
5. Tune hyperparameters on validation only.
6. Freeze pipeline.
7. Single final evaluation on test set.

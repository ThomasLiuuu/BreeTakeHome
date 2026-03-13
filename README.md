# Bree Take-Home: AI-Powered Loan Application Processor

ML model to replace a rule-based loan scoring system, with evaluation, explainability, and fairness analysis.

## Quick Start

```bash
pip install -r requirements.txt
python generate_data.py          # generates loan_applications.csv
jupyter notebook notebook.ipynb  # main analysis
```

## Repo Structure

```
.
├── README.md
├── requirements.txt
├── generate_data.py            # dataset generation (from assignment)
├── loan_applications.csv       # 2,000 applications
├── notebook.ipynb              # main analysis notebook
└── figures/                    # generated plots
```

## Approach

**Data handling:**
- Excluded 164 ongoing applications (no outcome). Checked that they're not systematically different from resolved apps.
- Missing `documented_monthly_income` (~17%) kept as-is with a binary `has_docs` flag - missingness is a risk signal.
- Class imbalance (2.4:1) handled via `class_weight='balanced'` / `scale_pos_weight`.

**Feature engineering:**
- `income_gap` - difference between stated and documented income (fraud signal)
- `income_loan_ratio` - can they afford this loan?
- `balance_loan_ratio`, `net_flow`, `spend_ratio` - account health
- `misrepresenting` flag for income gaps > 50%

**Models:** Logistic regression, random forest, XGBoost. Went with XGBoost (best recall) + SHAP for explainability.

## Results

| Metric | Rule-Based | XGBoost (Ours) |
|--------|:----------:|:--------------:|
| AUC-ROC | 0.72 | **0.75** |
| F1 (default) | 0.54 | **0.57** |
| FPR | 47.5% | **31.7%** |
| Recall | **78.0%** | 69.7% |

41 fewer good applicants wrongly denied (-15.8pp FPR), at cost of 9 more missed defaults (-8.3pp recall).

**Fairness:** Self-employed DI ratio goes from 0.62 (FAIL) to 0.99 (PASS).

## What I'd Do With More Time

- Survival analysis for ongoing applications
- Calibration analysis (are predicted probabilities accurate?)
- Cost-sensitive learning (missed default costs more than a false denial)
- Time-based train/test split instead of random
- Rule-based score as an additional feature
- Model monitoring pipeline with drift detection

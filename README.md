# Cell Nucleus Classification - ML Assignment 2

## a. Problem statement
Compare five classifiers for malignant versus benign breast-mass classification using cell-nucleus measurements. Educational experiment only.

## b. Dataset description
UCI Breast Cancer Wisconsin (Diagnostic): 569 instances, 30 numerical input features, no missing values. Target: 1=malignant (212), 0=benign (357). ID is excluded.
Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI. https://doi.org/10.24432/C5DW2B. License: CC BY 4.0.
The default loader uses sklearn's packaged copy; --data accepts original UCI wdbc.data.
Split: stratified 80/20, seed 42; 455 training and 114 test rows. Scaling occurs inside pipelines fitted only on training data. Five-fold CV uses training data only. Model settings are predefined, not optimized on the test set.

## c. Github Repository Link
TODO: Add your actual GitHub repository URL.
Live Streamlit app: TODO: Add your deployed app URL.

## d. Models used
Accuracy, AUC, precision, recall, F1 and MCC below use the identical held-out test set. Malignant is positive. AUC uses probability scores, not predicted labels. Precision/recall/F1 are binary, with zero_division=0.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations about model performance
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Held-out F1=0.9512, recall=0.9286, AUC=0.9960. Linear baseline; scaling supports optimization. Examine whether a linear boundary is sufficient. |
| Decision Tree | Held-out F1=0.8571, recall=0.7857, AUC=0.9471. Captures nonlinear rules; depth and leaf limits reduce overfitting. Compare training-only validation with held-out results. |
| KNN | Held-out F1=0.9383, recall=0.9048, AUC=0.9825. Distance-based classifier; scaling prevents large-scale features dominating distances. |
| Naive Bayes | Held-out F1=0.8889, recall=0.8571, AUC=0.9891. Gaussian feature likelihoods; correlated nucleus measurements may limit its conditional-independence assumption. |
| Support Vector Machine | Held-out F1=0.9630, recall=0.9286, AUC=0.9947. Scaled RBF kernel can model nonlinear boundaries; probabilities provide the AUC input. |
| Overall winner | Logistic Regression: selected by highest mean training-only 5-fold CV F1, with CV AUC as tie-breaker. Test scores are a separate final evaluation. |

### Training-only cross-validation
| ML Model Name | CV F1 | CV AUC |
| --- | --- | --- |
| Logistic Regression | 0.9640 | 0.9958 |
| Support Vector Machine | 0.9610 | 0.9949 |
| KNN | 0.9575 | 0.9860 |
| Naive Bayes | 0.9170 | 0.9870 |
| Decision Tree | 0.8963 | 0.9356 |

## Run locally and on BITS Lab
1. Install dependencies: `python -m pip install -r requirements.txt` (on first run, install numpy pandas scikit-learn matplotlib joblib streamlit).
2. Train: `python app.py --train`.
3. Start app: `python -m streamlit run app.py`.
4. Upload test_data.csv: all 30 features plus target, with 0/1 labels.
5. The app loads saved pipelines and evaluates all five models without fitting uploaded data.
Only load trusted locally generated joblib artifacts. Use the same Python minor version and dependency versions in lab/cloud as locally; retrain if versions change.

## Repository contents
app.py (training + app source), requirements.txt, README.md, test_data.csv, model/*.joblib, model/metadata.json, data/dataset.csv, results/metrics.csv, results/cross_validation.csv, confusion plots and classification reports.

## Streamlit Community Cloud
Push the required files to your GitHub repository. Create an app in Streamlit Community Cloud, choose the repository and branch, and set app.py as the main file. Configure a compatible Python version. Verify CSV upload, model selection, metrics and confusion matrix on the live URL.

## Final submission - action required
Execute the project on BITS Virtual Lab and capture your own screenshot. Review the code and replace these generated observations with your own interpretation. Add real GitHub and live app links. Create one PDF in this order: GitHub link, live app link, BITS execution screenshot, full finalized README content. Check both clickable links and submit by 8 November 2026, 23:59.
AI is a learning aid; understand and adapt the implementation and follow your institution's originality requirements. No lab screenshot, deployment or submission is performed by this script.

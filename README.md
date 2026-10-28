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
[GitHub Repository](https://github.com/sunithajaisonp08-alt/ML-Assignment-2)

Live Streamlit app: [Cell Nucleus Model Explorer](https://ml-assignment-2-dnxpr2bgeeh9xhgjdva2i4.streamlit.app/)

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
The following observations describe my results on the same 114-record held-out test set. Malignant is the positive class. These conclusions apply to this experiment and should not be treated as evidence of clinical performance.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | I observed an accuracy of 96.49% and the highest test AUC, 0.9960. Its recall of 0.9286 matched SVM, although its precision and F1 were slightly lower. This shows that a linear model provided a strong baseline for this dataset. It also achieved the highest mean training-only cross-validation F1, 0.9640, so it was the model selected by the predefined validation rule. |
| Decision Tree | I observed the lowest accuracy, 90.35%, and the lowest malignant recall, 0.7857, among the five models. On this test set, it missed 9 of the 42 malignant cases. Although its rules are relatively easy to interpret, this fitted tree was less effective than the other classifiers in identifying malignant cases. These scores alone do not establish whether the cause was overfitting or underfitting. |
| KNN | KNN achieved 95.61% accuracy, 0.9744 precision and 0.9383 F1. Its recall of 0.9048 means it identified 38 of the 42 malignant cases. I used scaling because KNN depends on distances between measurements. Its results were competitive, but its recall and F1 were below those of Logistic Regression and SVM on this split. |
| Naive Bayes | Gaussian Naive Bayes achieved 92.11% accuracy and 0.8889 F1, below Logistic Regression, KNN and SVM. Its AUC of 0.9891 was nevertheless high, showing that ranking performance and classification at the model's default decision rule can differ. Correlated nucleus measurements could affect its conditional-independence assumption, but this is a possible explanation rather than a cause established by these results. |
| Support Vector Machine | SVM achieved the highest test accuracy, 97.37%, F1, 0.9630, and MCC, 0.9442. Its precision of 1.0000 means it produced no false-positive malignant predictions on these 114 records. Recall was 0.9286, so it still missed 3 of the 42 malignant cases. I therefore consider it the strongest performer on the held-out test set by accuracy, F1 and MCC, without interpreting perfect precision as perfect overall classification. |
| Overall, winner for this dataset? | SVM was the winner on the held-out test set by accuracy, F1 and MCC. However, Logistic Regression was the model selected before test evaluation because its mean training-only five-fold CV F1 was higher (0.9640 versus 0.9610 for SVM). I distinguish the test-set winner from the validation-selected model; I did not use the test results to tune the models. |

### Training-only cross-validation
| ML Model Name | CV F1 | CV AUC |
| --- | --- | --- |
| Logistic Regression | 0.9640 | 0.9958 |
| Support Vector Machine | 0.9610 | 0.9949 |
| KNN | 0.9575 | 0.9860 |
| Naive Bayes | 0.9170 | 0.9870 |
| Decision Tree | 0.8963 | 0.9356 |

## Run locally and on BITS Lab
1. Install dependencies: `python -m pip install -r requirements.txt` (on first run, install numpy pandas scikit-learn matplotlib joblib streamlit). -- Completed
2. Train: `python app.py --train`. -- Completed
3. Start app: `python -m streamlit run app.py`. -- Completed
4. Upload test_data.csv: all 30 features plus target, with 0/1 labels. -- Completed
5. The app loads saved pipelines and evaluates all five models without fitting uploaded data. -- Completed
Only load trusted locally generated joblib artifacts. Use the same Python minor version and dependency versions in lab/cloud as locally; retrain if versions change. -- Completed

## Repository contents
app.py (training + app source), requirements.txt, README.md, test_data.csv, model/*.joblib, model/metadata.json, data/dataset.csv, results/metrics.csv, results/cross_validation.csv, confusion plots and classification reports.-- Completed

## Streamlit Community Cloud
The application has been deployed using `app.py` as the entry point and Python 3.13. The live link is included above. The application provides labeled test CSV upload, selection among all five saved classifiers, all six evaluation metrics, confusion matrices, classification reports and a comparison table for all five models. It does not retrain models on uploaded test data.

Before submission, verify CSV upload and each model on the live application, and confirm that both submission links open without account-owner access.

## BITS Virtual Lab execution
BITS Virtual Lab execution details added. 

**Preserve this README before running training:** `python app.py --train` regenerates `README.md`. Back up this finalized version outside the project folder and restore it after training, retaining any result changes that need to be reflected in the tables.

## Final submission checklist
- [ ] Verify that the GitHub repository is accessible and contains all required files. --Completed
- [ ] Verify that the live application accepts the test CSV and evaluates all five models.--Completed
- [ ] Complete BITS Virtual Lab execution and capture one genuine screenshot. --Completed
- [ ] Review the observations and ensure they reflect my understanding of the results. -- added Observations
- [ ] Prepare one PDF in this order: clickable GitHub link, clickable Streamlit link, BITS Lab screenshot, complete finalized README.
- [ ] Open the PDF and check link functionality, table readability and screenshot clarity.



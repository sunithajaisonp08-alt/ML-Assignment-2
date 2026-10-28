"""Learning companion for ML Assignment 2 (read and adapt before submission).

INSTALL: python -m pip install numpy pandas scikit-learn matplotlib joblib streamlit
TRAIN:   python app.py --train
APP:     python -m streamlit run app.py
JUPYTER: %run app.py --train

Uses sklearn's packaged copy of UCI Wisconsin Diagnostic Breast Cancer data.
Optionally train from the original UCI file: --train --data data/wdbc.data
All generated files live beside this script. Training overwrites generated outputs.
Run training on BITS Lab too and capture your own execution screenshot.
"""
from pathlib import Path
import argparse
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic'
SEED = 42
TARGET = 'target'
SLUGS = {'Logistic Regression': 'logistic_regression', 'Decision Tree': 'decision_tree',
         'KNN': 'knn', 'Naive Bayes': 'naive_bayes', 'Support Vector Machine': 'svm'}


def dataset(path=None):
    """Use consistent column names and malignant=1 for either data source."""
    reference = load_breast_cancer(as_frame=True)
    columns = list(reference.data.columns)
    if path:
        raw = pd.read_csv(path, header=None)
        if raw.shape != (569, 32):
            raise ValueError('Expected original wdbc.data: 569 rows, 32 columns, no header.')
        X = raw.iloc[:, 2:].copy()
        X.columns = columns
        y = raw.iloc[:, 1].map({'M': 1, 'B': 0})
    else:
        X = reference.data.copy()
        # sklearn's native mapping is malignant=0; explicitly reverse it.
        y = 1 - reference.target
    y = y.rename(TARGET)
    if y.isna().any() or X.isna().any().any():
        raise ValueError('Unexpected missing values or diagnosis labels.')
    assert X.shape[0] >= 500 and X.shape[1] >= 12
    return X, y


def models():
    # Each pipeline fits scaling only on its training partition, including in CV.
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True, random_state=SEED)),
    }


def evaluate(model, X, y):
    predictions = model.predict(X)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    values = {
        'Accuracy': accuracy_score(y, predictions),
        'AUC': roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else np.nan,
        'Precision': precision_score(y, predictions, zero_division=0),
        'Recall': recall_score(y, predictions, zero_division=0),
        'F1': f1_score(y, predictions, zero_division=0),
        'MCC': matthews_corrcoef(y, predictions),
    }
    return values, predictions, probabilities


def markdown_table(frame):
    # No optional tabulate dependency required.
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |'
        for row in frame.itertuples(index=False, name=None))


def train(data_path=None):
    X, y = dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    for folder in ['model', 'results', 'data']:
        (ROOT / folder).mkdir(exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(ROOT / 'data' / 'dataset.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, validation = [], []
    for name, model in models().items():
        # Compare predefined models on training-only CV; do not tune on test data.
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring={'f1': 'f1', 'auc': 'roc_auc'})
        validation.append({'ML Model Name': name, 'CV F1': scores['test_f1'].mean(), 'CV AUC': scores['test_auc'].mean()})
        model.fit(X_train, y_train)
        values, predicted, _ = evaluate(model, X_test, y_test)
        rows.append({'ML Model Name': name, **values})
        joblib.dump(model, ROOT / 'model' / f'{SLUGS[name]}.joblib')
        report = classification_report(y_test, predicted, labels=[0, 1], target_names=['Benign', 'Malignant'], zero_division=0, output_dict=True)
        (ROOT / 'results' / f'{SLUGS[name]}_report.json').write_text(json.dumps(report, indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, predicted, labels=[0, 1]), display_labels=['Benign', 'Malignant']).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(ROOT / 'results' / f'{SLUGS[name]}_confusion.png', dpi=150)
        plt.close(fig)
    metrics = pd.DataFrame(rows)
    cv_results = pd.DataFrame(validation).sort_values(['CV F1', 'CV AUC'], ascending=False)
    winner = cv_results.iloc[0]['ML Model Name']
    metrics.to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    cv_results.to_csv(ROOT / 'results' / 'cross_validation.csv', index=False)
    metadata = {'features': list(X.columns), 'target': TARGET, 'mapping': {'0': 'Benign', '1': 'Malignant'},
                'train_rows': len(X_train), 'test_rows': len(X_test), 'seed': SEED,
                'sklearn_version': sklearn.__version__, 'python_version': platform.python_version(),
                'selected_model': winner, 'source': SOURCE}
    (ROOT / 'model' / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    # Pin installed versions so cloud loads artifacts with matching sklearn.
    from importlib.metadata import version
    dependencies = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib', 'streamlit']
    lines = []
    for package in dependencies:
        try:
            lines.append(f'{package}=={version(package)}')
        except Exception:
            lines.append(package)
    (ROOT / 'requirements.txt').write_text('\n'.join(lines) + '\n')
    explanations = {
        'Logistic Regression': 'Linear baseline; scaling supports optimization. Examine whether a linear boundary is sufficient.',
        'Decision Tree': 'Captures nonlinear rules; depth and leaf limits reduce overfitting. Compare training-only validation with held-out results.',
        'KNN': 'Distance-based classifier; scaling prevents large-scale features dominating distances.',
        'Naive Bayes': 'Gaussian feature likelihoods; correlated nucleus measurements may limit its conditional-independence assumption.',
        'Support Vector Machine': 'Scaled RBF kernel can model nonlinear boundaries; probabilities provide the AUC input.'}
    observations = []
    for row in rows:
        observations.append({'ML Model Name': row['ML Model Name'], 'Observation about model performance':
            f"Held-out F1={row['F1']:.4f}, recall={row['Recall']:.4f}, AUC={row['AUC']:.4f}. " + explanations[row['ML Model Name']]})
    observations.append({'ML Model Name': 'Overall winner', 'Observation about model performance':
                         f'{winner}: selected by highest mean training-only 5-fold CV F1, with CV AUC as tie-breaker. Test scores are a separate final evaluation.'})
    readme = f'''# Cell Nucleus Classification - ML Assignment 2

## a. Problem statement
Compare five classifiers for malignant versus benign breast-mass classification using cell-nucleus measurements. Educational experiment only.

## b. Dataset description
UCI Breast Cancer Wisconsin (Diagnostic): 569 instances, 30 numerical input features, no missing values. Target: 1=malignant (212), 0=benign (357). ID is excluded.
Source: {SOURCE}
Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI. https://doi.org/10.24432/C5DW2B. License: CC BY 4.0.
The default loader uses sklearn's packaged copy; --data accepts original UCI wdbc.data.
Split: stratified 80/20, seed {SEED}; {len(X_train)} training and {len(X_test)} test rows. Scaling occurs inside pipelines fitted only on training data. Five-fold CV uses training data only. Model settings are predefined, not optimized on the test set.

## c. Github Repository Link
TODO: Add your actual GitHub repository URL.
Live Streamlit app: TODO: Add your deployed app URL.

## d. Models used
Accuracy, AUC, precision, recall, F1 and MCC below use the identical held-out test set. Malignant is positive. AUC uses probability scores, not predicted labels. Precision/recall/F1 are binary, with zero_division=0.

{markdown_table(metrics)}

### Observations about model performance
{markdown_table(pd.DataFrame(observations))}

### Training-only cross-validation
{markdown_table(cv_results)}

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
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nTraining-only CV selection: {winner}')
    print(f'Files saved beside script: {ROOT}')
    print('Next: review README and run python -m streamlit run app.py')


def app():
    import streamlit as st
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    st.set_page_config(page_title='Cell Nucleus Model Explorer', page_icon='🔬', layout='wide')
    st.title('Cell Nucleus Model Explorer')
    st.caption('Five classifiers · one held-out dataset · malignant = positive class')
    metadata_path = ROOT / 'model' / 'metadata.json'
    if not metadata_path.exists():
        st.info('Run python app.py --train first, then restart this app.')
        st.stop()
    metadata = json.loads(metadata_path.read_text())
    if metadata['sklearn_version'] != sklearn.__version__:
        st.error('scikit-learn version differs from training. Install the generated requirements or retrain with this environment.')
        st.stop()
    st.sidebar.header('Evaluation controls')
    selected = st.sidebar.selectbox('Classifier', list(SLUGS))
    upload = st.sidebar.file_uploader('Upload labeled test CSV', type=['csv'])
    st.sidebar.caption('Required: 30 measurement columns and target (0=benign, 1=malignant). No retraining occurs.')
    st.sidebar.download_button('Download example test CSV', (ROOT / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
    st.info('Educational model comparison. Results describe this dataset and split.')
    try:
        frame = pd.read_csv(upload) if upload is not None else pd.read_csv(ROOT / 'test_data.csv')
        required = metadata['features'] + [TARGET]
        if len(frame) == 0 or len(frame) > 10000:
            raise ValueError('Provide between 1 and 10,000 test records.')
        if set(frame.columns) != set(required) or len(frame.columns) != len(required):
            raise ValueError('CSV must contain exactly the feature columns in the example CSV plus target; no ID or index column.')
        frame = frame[required].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError('Missing or infinite values are not accepted.')
        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError('target must contain only 0 (benign) or 1 (malignant).')
        X, y = frame[metadata['features']], frame[TARGET].astype(int)
        results, details = [], {}
        for name, slug in SLUGS.items():
            saved = joblib.load(ROOT / 'model' / f'{slug}.joblib')
            values, predicted, probabilities = evaluate(saved, X, y)
            results.append({'ML Model Name': name, **values})
            details[name] = (predicted, probabilities)
    except Exception as error:
        st.error(f'Cannot evaluate: {error}')
        st.stop()
    st.write(f'Evaluating **{len(frame)} records** from ' + ('your uploaded CSV.' if upload is not None else 'the saved held-out test set.'))
    if y.nunique() < 2:
        st.warning('AUC is undefined because this CSV contains only one target class.')
    comparison = pd.DataFrame(results).set_index('ML Model Name')
    st.subheader(selected)
    for column, (metric, value) in zip(st.columns(6), comparison.loc[selected].items()):
        column.metric(metric, 'N/A' if pd.isna(value) else f'{value:.4f}')
    left, right = st.columns(2)
    predictions, probabilities = details[selected]
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, predictions, labels=[0, 1]), display_labels=['Benign', 'Malignant']).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('Actual versus predicted diagnosis')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader('Classification report')
        report = classification_report(y, predictions, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.subheader('All five models on the same test data')
    st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), use_container_width=True)
    st.download_button('Download comparison metrics', comparison.to_csv().encode(), 'comparison_metrics.csv', 'text/csv')
    exported = frame.copy()
    exported['predicted_target'] = predictions
    exported['malignant_probability'] = probabilities
    st.download_button('Download selected model predictions', exported.to_csv(index=False).encode(), 'predictions.csv', 'text/csv')
    with st.expander('Dataset, method and model selection'):
        st.write(f"Source: {SOURCE}")
        st.write(f"Training rows: {metadata['train_rows']}; held-out rows: {metadata['test_rows']}; split seed: {SEED}.")
        st.write(f"Training-only 5-fold CV F1 selection: {metadata['selected_model']}. Uploaded data is evaluated only; it is never used to fit models.")
        st.dataframe(frame.head())


if __name__ == '__main__':
    if '--train' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--data', type=Path, help='Optional original UCI wdbc.data')
        args = parser.parse_args()
        train(args.data)
    else:
        app()

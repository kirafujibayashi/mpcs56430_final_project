import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

pd_data = pd.read_csv('/Users/kirafujibayashi/Library/Mobile Documents/com~apple~CloudDocs/Documents/UChicago/Scientific Computing/Assignments/Final Project/data/pd_speech_features.csv')

# Define feature subsets
baseline_columns = [
    'PPE', 'DFA', 'RPDE', 'numPulses', 'numPeriodsPulses',
    'meanPeriodPulses', 'stdDevPeriodPulses', 'locPctJitter', 'locAbsJitter',
    'rapJitter', 'ppq5Jitter', 'ddpJitter', 'locShimmer', 'locDbShimmer',
    'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
    'meanAutoCorrHarmonicity', 'meanNoiseToHarmHarmonicity', 'meanHarmToNoiseHarmonicity'
]

mfcc_columns = [col for col in pd_data.columns if 'MFCC' in col]

wavelet_columns = [
    col for col in pd_data.columns if col.startswith('tqwt_') and '_dec_' in col
]

vocal_fold_columns = [
    col for col in pd_data.columns if any(keyword in col for keyword in [
        'tqwt_energy', 'tqwt_entropy_shannon', 'tqwt_kurtosisValue', 'tqwt_skewnessValue'
    ])
]

# Prepare feature subsets
X_baseline = pd_data[baseline_columns]
X_mfcc = pd_data[mfcc_columns]
X_wavelet = pd_data[wavelet_columns]
X_vocal_fold = pd_data[vocal_fold_columns]
y = pd_data['class']

# Define classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'k-NN': KNeighborsClassifier(),
    'Multilayer Perceptron': MLPClassifier(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Logistic Regression (with PCA)': LogisticRegression(max_iter=1000, random_state=42)
}

# Helper function for PCA transformation
def apply_pca(X, variance_threshold=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=variance_threshold)  # Retain components explaining 95% variance
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_

# Evaluate classifiers on each feature subset
results = {}
feature_subsets = {
    'Baseline': X_baseline,
    'MFCC': X_mfcc,
    'Wavelet': X_wavelet,
    'Vocal Fold': X_vocal_fold
}

for subset_name, X_subset in feature_subsets.items():
    results[subset_name] = {}
    for clf_name, clf in classifiers.items():
        if clf_name == 'Logistic Regression (with PCA)':
            # Apply PCA for Logistic Regression
            X_pca, variance_ratio = apply_pca(X_subset)
            X_to_use = X_pca
        else:
            X_to_use = X_subset
        acc = cross_val_score(clf, X_to_use, y, cv=5, scoring='accuracy').mean()
        f1 = cross_val_score(clf, X_to_use, y, cv=5, scoring='f1').mean()
        results[subset_name][clf_name] = {
            'Accuracy': acc,
            'F1-Score': f1
        }

# Format results into a DataFrame
final_results = []
for subset, clf_results in results.items():
    for clf_name, metrics in clf_results.items():
        final_results.append({
            'Feature Subset': subset,
            'Classifier': clf_name,
            'Accuracy': metrics['Accuracy'],
            'F1-Score': metrics['F1-Score']
        })

# Convert results to DataFrame
results_df = pd.DataFrame(final_results)

# Save the results to a CSV file
results_df.to_csv("classifier_results_with_pca_2.csv", index=False)

print("Results saved to 'classifier_results_with_pca_2.csv'")

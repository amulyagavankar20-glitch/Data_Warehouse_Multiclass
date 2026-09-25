# 🏥 Multi-Class Disease Detection & Advanced Anomaly Detection (Early Warning System)

An advanced, production-grade clinical intelligence and predictive modeling framework designed to detect multi-class disease sublabels and provide early-stage sub-clinical risk warnings through unsupervised anomaly detection.

---

## 🌟 Project Architecture & Core Pillars

This repository is structured around a **4-Core Methodological Framework** bridging classic data science storytelling with state-of-the-art AIML anomaly detection:

### Phase I: Data Engineering & Prevention of Data Leakage
* **Clinical Data Hygiene:** Cleaned duplicate hospital registry logs and safely parsed multi-class comma-delimited disease indicator flags into structured binary columns.
* **Leakage-Free Preprocessing:** Performed stratified splitting upfront to preserve rare disease class proportions. All scaling parameters (`StandardScaler`) are derived exclusively from the training population to prevent test set data leakage.

### Phase II: Unsupervised Baseline Profiling (The Anomaly Engine)
* **The Clinical Premise:** True early-warning systems must identify deviation from health rather than relying solely on known disease tags.
* **Model Selection:** Trained **Isolation Forest**, **One-Class SVM**, and **Deep Autoencoders** *strictly* on healthy baseline patients (`label == 'Normal'`) to learn the compressed latent distribution of a normal physiological profile.

### Phase III: Sub-Clinical Validation & Imbalance Handling
* **Avoiding SMOTE (Clinical Integrity):** Artificial synthesis of clinical biomarkers (e.g., blood pressure, glucose, HbA1c) can violate human biological laws. Class imbalance is handled entirely via stratified sampling, cost-sensitive learning weights, and reconstruction error analysis.
* **Early Warning Audit:** Evaluated whether high reconstruction errors or anomaly scores appear in patients marked with early-stage combinations (e.g., hypertension `HY`, diabetes `DI`) before full-blown clinical manifestation.

### Phase IV: Hybrid Meta-Feature Supervised Classifier
* **Closing the Loop:** Unsupervised anomaly outputs (Isolation Forest decision functions and Autoencoder MSE) are treated as powerful **meta-features**.
* **Final Multi-Class Prediction:** Meta-features are concatenated with original biomarkers and fed into an ensemble classifier (`RandomForestClassifier` with balanced class weights) to accurately predict multi-class disease sublabels.

---

## 📊 Empirical Results & Early Warning Audit

When tested on the independent test split ($\approx 55,700$ records), our unsupervised models successfully captured disease severity gradients:

| Disease Sublabel | Record Count | Mean Isolation Forest Score (Lower = More Abnormal) | Mean Autoencoder MSE (Higher = More Abnormal) | Clinical Inference |
| :--- | :--- | :--- | :--- | :--- |
| **`DI_HY`** (Diabetes + Hypertension) | 13,381 | **0.0035** | 0.1031 | Highest structural deviation (severe multi-morbidity) |
| **`DI`** (Diabetes Only) | 6,390 | 0.0213 | 0.0674 | High anomaly score |
| **`HY`** (Hypertension Only) | 13,829 | 0.0283 | 0.0570 | High anomaly score |
| **`DI_HT_HY`** (Triple Co-morbidity) | 181 | 0.0341 | **0.3778** | Extreme reconstruction error |
| **`N`** (Healthy Baseline Normals) | 20,930 | **0.1060** | **0.0034** | Safely recognized as normal baseline |

The hybrid meta-feature augmented **Random Forest Classifier** achieved near-perfect multi-class classification metrics ($1.00$ Precision, Recall, and F1-Score) across all target diagnostic classes.

---

## ⚙️ Tech Stack & Requirements
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Isolation Forest, One-Class SVM, Random Forest)
* **Deep Learning:** TensorFlow / Keras (Deep Autoencoders)
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/disease-anomaly-early-warning.git](https://github.com/your-username/disease-anomaly-early-warning.git)
   cd disease-anomaly-early-warning
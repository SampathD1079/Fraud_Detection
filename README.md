# Fraud_Detection
# Invoice Fraud Detection System

An intelligent invoice fraud detection system that combines **PDF forensic analysis, rule-based verification, and machine learning** to identify potentially fraudulent invoices.

The system first analyzes the internal structure of PDF invoices, including metadata, PDF version, cross-reference structures, trailer IDs, content streams, fonts, and document modification information. These forensic indicators are converted into a rule-based fraud score.

A **Random Forest Classifier** is then trained separately using labeled invoice data containing forensic features and fraud labels. During invoice processing, the trained model generates a fraud probability, which is converted into an ML score and combined with the existing rule-based score to produce a final risk score.

## Key Features

* PDF forensic analysis and structural inspection
* Metadata and document modification analysis
* Font and content-stream anomaly detection
* Rule-based fraud scoring
* Random Forest-based fraud probability prediction
* Combined rule-based + ML risk scoring
* Risk classification into Low Risk, Suspicious, and High Risk/Fraud
* Vendor and invoice number extraction
* Duplicate invoice detection
* SQLite database storage
* Batch processing of multiple PDF invoices
* Separate ML training and inference pipelines

## Machine Learning

The ML component uses a **Random Forest Classifier** trained on labeled invoice forensic features:

* `creator`
* `modification_creation`
* `pdf_version`
* `trailer_id_match`
* `match_xref`
* `font_score`

The training pipeline evaluates the model using accuracy, precision, recall, F1-score, ROC-AUC, classification report, and confusion matrix before saving the trained model for inference.

The trained model is loaded during invoice processing and produces a fraud probability. This probability is converted into an ML score and combined with the existing rule-based score to generate the final risk score.

## Risk Classification

The final score is categorized as:

* **Low Risk / Normal** — score below 40
* **Suspicious** — score between 40 and 69
* **High Risk / Fraud** — score 70 or above

The system treats the result as a **risk estimate rather than an absolute fraud verdict**.

## Database

Invoice information and fraud-analysis results are stored in SQLite, including:

* Invoice number
* Vendor name
* Rule-based score
* ML probability
* ML score
* Final score
* Risk classification
* Creation timestamp

## Tech Stack

**Python | Scikit-learn | Random Forest | Pandas | PyPDF2 | pikepdf | SQLite | Joblib | NumPy**

## Architecture

PDF Invoice → Text & Forensic Analysis → Rule-Based Features → Random Forest ML Model → ML Probability → Combined Risk Score → Fraud Classification → Database

The ML training process is intentionally separated from invoice processing. The model is trained explicitly using labeled data and is not retrained automatically while processing invoices.

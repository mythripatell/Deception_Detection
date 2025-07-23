# Deception_Detection

This project implements a supervised machine learning pipeline to detect deceptive news articles using **Natural Language Processing (NLP)** and **Logistic Regression**. It transforms raw news content into numerical features using **TF-IDF vectorization**, allowing accurate classification of articles as *real* or *deceptive*. The model achieved an accuracy of **87.5%** on the test dataset.


## Tech Stack

- **Programming Language:** Python  
- **Libraries:** Scikit-learn, NLTK, NumPy, Matplotlib, Seaborn  
- **Feature Extraction:** TF-IDF  
- **Model Used:** Logistic Regression  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC

---

## Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 87.5%   |
| Precision | 0.89    |
| Recall    | 0.85    |
| F1-Score  | 0.87    |

---

## Visual Outputs

### Confusion Matrix  
<img src="images/confusion_matrix.png" alt="Confusion Matrix" width="400"/>

### ROC Curve  
<img src="images/roc_curve.png" alt="ROC Curve" width="400"/>

---

## Objective

To build an interpretable and scalable machine learning solution for detecting misinformation in digital news content. This system addresses the increasing challenge of media deception using classical yet effective NLP and classification techniques.

---

## 📁 Project Structure
deception-detection/
├── images/
│ ├── confusion_matrix.png
│ └── roc_curve.png
├── data/
├── src/
│ ├── preprocessing.py
│ └── train_model.py
├── model/
├── requirements.txt
└── README.md


---

## 📜 License

This project is licensed under the MIT License.



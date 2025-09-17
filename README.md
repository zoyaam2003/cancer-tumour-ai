# Machine Learning for Cancer Diagnosis and Health Applications

This project investigates the application of supervised machine learning and deep learning models for medical diagnostics.  
It focuses on two case studies: classification of **breast tumours** using tabular fine needle aspirate (FNA) data, and classification of **brain tumour MRI images** into multiple categories.  
The aim is to demonstrate the potential of AI to support early detection while addressing ethical considerations such as bias, privacy, and clinical safety.

Developed as part of an applied machine learning project, the study benchmarks classical ML algorithms (e.g. Logistic Regression, Random Forest, SGDClassifier) against deep learning models (e.g. CNNs, MobileNetV2), and evaluates them for both predictive performance and ethical robustness.

---

## Project Objectives
- Classify breast tumours as benign or malignant using tabular diagnostic features.  
- Develop and benchmark deep learning models for multi-class brain tumour detection from MRI scans.  
- Compare performance across models using healthcare-suitable metrics (accuracy, recall, F1-score, ROC-AUC).  
- Evaluate the impact of model design on interpretability and clinical relevance.  
- Address ethical considerations, including fairness, data privacy (GDPR/HIPAA), and risks of misclassification in clinical settings.  

---

## Project Structure
ml_cancer_diagnosis/

MachineLearningReport.ipynb : Code for machine learning project.
AI Report.docx : Final written report including results, discussion, and ethical analysis
README.md: Project overview and documentation

---

## Models Implemented
| Model              | Type             | Description |
|--------------------|------------------|-------------|
| Logistic Regression | Classical ML     | Baseline classifier for breast tumour data |
| SGDClassifier      | Classical ML     | Linear classifier with stochastic gradient descent |
| Random Forest      | Classical ML     | Non-linear ensemble model for comparison |
| Custom CNN         | Deep Learning    | Convolutional neural network for MRI image classification |
| MobileNetV2        | Transfer Learning| Pre-trained CNN fine-tuned for brain tumour data |

---

## Project Results

**Breast Cancer Classification**  
- Logistic Regression achieved the highest accuracy (**97.5%**) with strong interpretability.  
- SGDClassifier delivered similar performance with computational efficiency.  
- Random Forest performed slightly lower but offered robustness.  

**Brain Tumour Classification**  
- Custom CNN achieved **~85% accuracy** with excellent recall for the “No tumour” class.  
- MobileNetV2 achieved **~82% accuracy**, showing the potential of transfer learning.  

Both tasks highlight the promise of AI for clinical diagnostics, though further work is required to address fairness, dataset diversity, and real-world generalisability.

---

## Data Sources
- **Breast Cancer Wisconsin (Diagnostic) dataset** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Brain MRI dataset** – [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  

*(Note: raw medical images are not redistributed here due to licensing restrictions.)*

---

## Tools & Libraries Used
- Python 3.10  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib / Seaborn  
- Pandas / NumPy  

---

## Citation
**Aamir, Zoya (2025).** *Machine Learning for Cancer Diagnosis: Breast Cancer and Brain Tumour Classification with Ethical Considerations*. Applied Machine Learning Project.  

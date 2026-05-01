## Credit Card Fraud Detection System

##  Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques.  
It addresses the challenge of **highly imbalanced data** and builds a robust classification model to identify fraud effectively.

---

##  Problem Statement
Credit card fraud is a critical issue in banking and fintech industries.  
The dataset used in this project contains a very small percentage of fraudulent transactions (~0.17%), making it a **class imbalance problem**.

---

##  Solution Approach
The project implements an end-to-end ML pipeline:

1. Data Loading and Exploration (EDA)  
2. Handling missing values  
3. Handling class imbalance using **SMOTE**  
4. Splitting data into training and testing sets  
5. Training model using **Random Forest Classifier**  
6. Evaluating model using:
   - Classification Report  
   - Confusion Matrix  
7. Visualizing results using graphs  

---

##  Key Features
- Handles imbalanced dataset using SMOTE  
- Implements supervised classification model  
- Provides performance evaluation metrics  
- Generates visual insights (graphs)  
- Saves trained model and outputs  

---

##  Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib  
- Seaborn  

---

##  Results

###  Fraud Distribution
![Fraud Distribution](images/fraud_distribution.png)

###  Transaction Amount Distribution
![Amount Distribution](images/amount_distribution.png)

###  Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

---

##  Model Performance
- Accuracy: ~99%  
- High recall for fraud detection  
- Reduced false negatives using imbalance handling  

---

##  Project Structure
Credit-Card-Fraud-Detection/
├── data/ # Dataset
├── models/ # Saved ML models
├── outputs/ # Reports and results
├── images/ # Visualizations
├── main.py # Main pipeline
├── README.md
├── requirements.txt

##  How to Run

1. Clone the repository:
git clone https://github.com/ayushkrdubey-23/Credit-Card-Fraud-Detection

2. Navigate to project folder:
cd Credit-Card-Fraud-Detection

3. Install dependencies:
pip install -r requirements.txt

4. Run the project:
python main.py

---

##  Key Learnings
- Handling imbalanced datasets is crucial in real-world ML problems  
- Evaluation metrics like confusion matrix are more important than accuracy in fraud detection  
- Visualization helps in understanding data patterns effectively  

---

##  Future Improvements
- Implement XGBoost for better performance  
- Add Precision-Recall curve  
- Deploy model using FastAPI  
- Build interactive dashboard  

---

##  Acknowledgment
This project was completed under the guidance of **Mr. Umesh Yadav (IIP)** in collaboration with **EDC IIT Delhi**.

---

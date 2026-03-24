# 🩺 Disease Prediction System
> AI-Powered Disease Detection | Powered by Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![SDG3](https://img.shields.io/badge/SDG--3-Good%20Health-brightgreen?style=for-the-badge)

---

## 📌 Project Overview
Yeh ek **AI-powered Disease Prediction System** hai jo
**Machine Learning (Random Forest)** use karke aapke
symptoms se possible bimari detect karta hai.

Banaya gaya hai **SDG-3 — Good Health & Well-Being** ke
liye taaki log apni sehat ko aasaani se samajh sakein.

---

## ✨ Features
| Feature | Detail |
|---------|--------|
| 🦠 Diseases | 16 bimariyan predict karta hai |
| 📋 Symptoms | 15 symptoms choose kar sakte ho |
| 🎯 Confidence | AI kitna sure hai % mein dikhata hai |
| 🌐 Language | Hindi + English dono mein |
| 📊 Charts | Interactive Plotly graphs |
| ⚡ Severity | Mild / Moderate / High / Critical |
| 🤖 ML Model | Random Forest (200 trees) |

---

## 🗂️ Project Files
Disease_Project/
├── app.py            → Main Streamlit Dashboard
├── disease.csv       → Dataset (symptoms + diseases)
├── model.py          → ML Model training script
├── requirements.txt  → Python libraries
└── README.md         → Ye file

---

## 🚀 Kaise Chalayein?

Step 1 — Libraries install karo
pip install -r requirements.txt

Step 2 — App chalao
streamlit run app.py

Step 3 — Browser mein dekho
http://localhost:8501

---

## 🦠 Diseases Covered
| # | Disease | Severity |
|---|---------|----------|
| 1 | Flu | 🟡 Moderate |
| 2 | Cold | 🟢 Mild |
| 3 | Dengue | 🔴 High |
| 4 | Malaria | 🔴 High |
| 5 | Typhoid | 🔴 High |
| 6 | Pneumonia | 🔴 High |
| 7 | Hepatitis | 🔴 High |
| 8 | Hypertension | 🔴 High |
| 9 | Heart Disease | 🚨 Critical |
| 10 | Migraine | 🟡 Moderate |
| 11 | Chickenpox | 🟡 Moderate |
| 12 | Bronchitis | 🟡 Moderate |
| 13 | Food Poisoning | 🟡 Moderate |
| 14 | Strep Throat | 🟡 Moderate |
| 15 | Anemia | 🟡 Moderate |
| 16 | Gastroenteritis | 🟢 Mild |

---

## 🧠 ML Model Details
| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 200 Estimators |
| Features | 15 Symptoms |
| Classes | 16 Diseases |
| Library | scikit-learn |

---

## 🌍 SDG-3 Connection
This project supports UN Sustainable Development Goal 3
Good Health and Well-Being.

AI technology ko use karke common logo tak health
awareness pahunchana is project ka main maqsad hai.

---

## ⚠️ Disclaimer
Yeh tool sirf educational purpose ke liye hai.
Kisi bhi bimari ke liye please asli doctor se milo.
Yeh medical advice nahi hai.

---

## 👨‍💻 Made with ❤️ for SDG-3 — Good Health & Well-Being
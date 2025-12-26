# 🧠 Bayesian Network–Based Medicine Recommendation System

This project implements a **Bayesian Network–based machine learning system** to recommend medicines based on selected symptoms. It uses probabilistic reasoning to model the relationship between **symptoms, diseases, and medicines**, and provides results through a simple graphical user interface (GUI).

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Features
- Bayesian Network model 
- Symptom → Disease → Medicine inference
- Probabilistic reasoning under uncertainty
- Tkinter-based interactive GUI
- Prediction history logging with timestamps
- Easy-to-extend medical dataset



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 🏗️ Model Architecture

The Bayesian Network structure used in this project is:
Symptom → Disease → Medicine
- Nodes represent medical variables  
- Directed edges represent conditional dependencies  
- Inference is performed using **Variable Elimination**

## 📂 Dataset Description

The dataset is stored in `data2.csv` and contains the following columns:

| Column Name    | Description |
|---------------|------------|
| symptoms      | Comma-separated list of symptoms |
| disease       | Associated disease |
| medicine      | Recommended medicine |
| uses          | Purpose of the medicine |
| dosage        | Dosage instructions |
| side_effects  | Possible side effects |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## ⚙️ Technologies Used
- Python
- Tkinter
- Pandas
- pgmpy (Bayesian Network modeling)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 🚀 How It Works
1. User selects one or more symptoms from the GUI  
2. Bayesian Network infers probable diseases  
3. The most likely medicine is predicted  
4. Medicine details (uses, dosage, side effects) are displayed  
5. Prediction history is saved in a log file  

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


1. Install required libraries:
   ```bash
pip install pandas pgmpy
2. Run the application: python main.py

Note: Ensure that data2.csv is located in the same directory as the Python file.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Output

1. Recommended medicine

2. Disease name

3. Uses and dosage

4. Side effects

5. History stored in history.txt
   
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚠️ Limitations

- Limited by dataset accuracy and medical scope

- Not a replacement for professional diagnosis

🔮 Future Improvements

- Expand dataset with more diseases and medicines

- Improve Bayesian Network structure learning

- Enhance GUI design

- Web or mobile application integration


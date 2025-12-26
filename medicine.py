import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import os
import datetime

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# ---------------- LOAD DATA ----------------
def load_data(filename="data2.csv"):
    try:
        df = pd.read_csv(filename)
        df["symptoms"] = df["symptoms"].apply(
            lambda x: [s.strip().lower() for s in x.split(",")]
        )
        return df
    except FileNotFoundError:
        messagebox.showerror("Error", f"{filename} not found!")
        return pd.DataFrame()


# ---------------- BAYESIAN NETWORK ----------------
def prepare_bn_data(df):
    rows = []
    for _, row in df.iterrows():
        for symptom in row["symptoms"]:
            rows.append({
                "symptom": symptom,
                "disease": row["disease"],
                "medicine": row["medicine"]
            })
    return pd.DataFrame(rows)


def build_bayesian_network(df):
    model = DiscreteBayesianNetwork([
        ("symptom", "disease"),
        ("disease", "medicine")
    ])
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model


def predict_medicine(selected_symptoms, model):
    infer = VariableElimination(model)
    predictions = []

    for symptom in selected_symptoms:
        try:
            result = infer.map_query(
                variables=["medicine"],
                evidence={"symptom": symptom}
            )
            predictions.append(result["medicine"])
        except:
            pass

    if not predictions:
        raise Exception("No matching medicine found.")

    return max(set(predictions), key=predictions.count)


# ---------------- UTILITIES ----------------
def get_all_symptoms(df):
    s = set()
    for row in df["symptoms"]:
        s.update(row)
    return sorted(s)


def log_history(symptoms, medicine):
    with open("history.txt", "a", encoding="utf-8") as f:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{time} | {', '.join(symptoms)} -> {medicine}\n")



def load_history():
    if not os.path.exists("history.txt"):
        return "No history available."
    with open("history.txt", "r") as f:
        return f.read()


# ---------------- GUI ACTIONS ----------------
def on_submit():
    selected = [s for s, v in symptom_vars.items() if v.get()]
    if not selected:
        messagebox.showwarning("Input Error", "Please select at least one symptom.")
        return

    try:
        medicine = predict_medicine(selected, bn_model)
        row = df[df["medicine"] == medicine].iloc[0]

        log_history(selected, medicine)

        output_text.config(state="normal")
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Recommended Medicine: {medicine}\n\n")
        output_text.insert(tk.END, f"Disease: {row['disease']}\n")
        output_text.insert(tk.END, f"Uses: {row['uses']}\n")
        output_text.insert(tk.END, f"Dosage: {row['dosage']}\n")
        output_text.insert(tk.END, f"Side Effects: {row['side_effects']}\n")
        output_text.config(state="disabled")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def show_history():
    win = tk.Toplevel(root)
    win.title("Medical History")
    win.geometry("500x400")
    txt = scrolledtext.ScrolledText(win)
    txt.pack(fill=tk.BOTH, expand=True)
    txt.insert(tk.END, load_history())
    txt.config(state="disabled")


def reset_selection():
    for v in symptom_vars.values():
        v.set(False)
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    output_text.config(state="disabled")


# ---------------- INITIALIZATION ----------------
df = load_data()
if df.empty:
    exit()

bn_df = prepare_bn_data(df)
bn_model = build_bayesian_network(bn_df)
symptoms_list = get_all_symptoms(df)


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Bayesian Network Medicine Recommendation System")
root.geometry("1000x700")

title = tk.Label(
    root,
    text="AI Medicine Recommendation System (Bayesian Network)",
    font=("Helvetica", 18, "bold")
)
title.pack(pady=20)

frame = tk.Frame(root)
frame.pack()

canvas = tk.Canvas(frame, width=650, height=400)
scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas)

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

symptom_vars = {}
for i, symptom in enumerate(symptoms_list):
    var = tk.BooleanVar()
    chk = tk.Checkbutton(scroll_frame, text=symptom.capitalize(), variable=var)
    chk.grid(row=i // 3, column=i % 3, sticky="w", padx=10)
    symptom_vars[symptom] = var

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="Submit", width=15, command=on_submit).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Reset", width=15, command=reset_selection).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="History", width=15, command=show_history).grid(row=0, column=2, padx=5)
tk.Button(btn_frame, text="Exit", width=15, command=root.destroy).grid(row=0, column=3, padx=5)

output_text = scrolledtext.ScrolledText(root, height=10, font=("Helvetica", 12))
output_text.pack(fill=tk.BOTH, padx=20, pady=10)
output_text.config(state="disabled")

root.mainloop()

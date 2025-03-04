import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
final_model = joblib.load("final_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names (same as in training)
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]

# Function to make predictions
def predict_heart_disease():
    try:
        # Get values from input fields
        patient_data = [float(entries[feature].get()) for feature in feature_names]

        # Convert to 2D NumPy array and scale
        # Convert to DataFrame before scaling to retain feature names
        new_patient_df = pd.DataFrame([patient_data], columns=feature_names)

        # Scale the new data
        new_patient_scaled = scaler.transform(new_patient_df)

        # Make predictions
        prediction_proba = final_model.predict_proba(new_patient_scaled)[0]
        prediction_class = final_model.predict(new_patient_scaled)[0]

        # Display result
        risk_percentage = prediction_proba[1] * 100
        result_text = f"Heart Disease Risk: {risk_percentage:.2f}%\n"
        result_text += f"Prediction: {'Heart Disease' if prediction_class == 1 else 'No Heart Disease'}"

        messagebox.showinfo("Prediction Result", result_text)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Create GUI window
root = tk.Tk()
root.title("Heart Disease Prediction")

# Create input fields
entries = {}
for i, feature in enumerate(feature_names):
    tk.Label(root, text=feature).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[feature] = entry

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease)
predict_button.grid(row=len(feature_names), columnspan=2, pady=10)

# Run the GUI application
root.mainloop()

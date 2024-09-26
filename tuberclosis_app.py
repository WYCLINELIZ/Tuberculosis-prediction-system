import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample data (replace with your actual dataset)
data = {
    'age': [25, 35, 45, 55, 28, 32, 48, 50, 30, 28],
    'weight': [70, 60, 80, 75, 65, 55, 90, 68, 75, 60],
    'height': [175, 160, 180, 165, 170, 155, 185, 163, 180, 160],
    'bmi': [22.8, 23.4, 24.7, 27.5, 22.5, 21.6, 26.3, 25.5, 23.1, 23.4],
    'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'cd4_Count': [500, 450, 550, 600, 520, 480, 560, 590, 510, 500],
    'outcome': ['Cured', 'Treated', 'Died', 'Cured', 'Treated', 'Died', 'Cured', 'Treated', 'Died', 'Cured']
}
df = pd.DataFrame(data)

# Preprocess data
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['outcome'] = df['outcome'].map({'Cured': 0, 'Treated': 1, 'Died': 2})

# Prepare data
X = df[['age', 'weight', 'height', 'bmi', 'sex', 'cd4_Count']]
y = df['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title("Treatment Outcome Prediction")

st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
height = st.sidebar.number_input("Height (cm)", min_value=140, max_value=220, value=170)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
sex = st.sidebar.radio("Sex", ['Male', 'Female'])
cd4_count = st.sidebar.number_input("CD4 Count", min_value=0, max_value=2000, value=500)


if st.button("Predict Outcome"):
    # Convert sex to numerical
    sex_num = 1 if sex == 'Male' else 0

    input_data = pd.DataFrame({
        'age': [age],
        'weight': [weight],
        'height': [height],
        'bmi': [bmi],
        'sex': [sex_num],
        'cd4_Count': [cd4_count]
    })
    prediction = model.predict(input_data)[0]

    outcome_map = {0: 'Cured', 1: 'Treated', 2: 'Died'}
    result = outcome_map[prediction]

    st.write(f"Predicted Outcome: **{result}**")

# 🚗 Used Car Price Prediction

This is a machine learning project I built to predict the estimated price of a used car based on details like its manufacturing year, brand, model, fuel type, kilometres travelled, and city.

I wanted to understand how machine learning works in a real project, from working with a dataset to training models and building a simple web app where users can try their own inputs.

## ✨ Features

- Predicts the estimated price of a used car.
- Uses two machine learning models: Linear Regression and Random Forest.
- Allows users to enter car details through a Streamlit interface.
- Shows predictions from both models for comparison.
- Includes basic dataset statistics and sample car listings.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- SciPy
- Joblib
- Streamlit

## 🤖 Machine Learning Models

I trained and compared two regression models:

- **Linear Regression** — used as the baseline model.
- **Random Forest Regressor** — used to capture more complex patterns in the data.

On my current test split, Random Forest achieved an R² score of approximately **0.723**, while Linear Regression achieved approximately **0.621**.

These results are based on my dataset and test split, so predictions may not always match actual market prices.

## 📂 Project Structure

```text
used-car-price-prediction/
├── app.py
├── train.py
├── eda.py
├── requirements.txt
├── encoder.pkl
├── linear_model.pkl
├── random_forest_model.pkl
└── data/
    └── UsedCars_Combined.csv
```

## 🚀 How to Run the Project

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd used-car-price-prediction
```

**2. Install the required packages**

```bash
pip install -r requirements.txt
```

**3. Start the Streamlit app**

```bash
streamlit run app.py
```

The app will open in your browser. Enter the car details and click the prediction button to see the estimated prices.

## 📌 What I Learned

While building this project, I learned how to:

- Load and explore a real-world dataset using Pandas.
- Prepare numerical and categorical features for machine learning.
- Use OneHotEncoder to convert categorical data into numerical features.
- Train and evaluate regression models.
- Compare model performance using MAE and R² score.
- Save trained models using Joblib.
- Connect saved models to a Streamlit application.

## ⚠️ Disclaimer

This project provides estimated prices for learning and demonstration purposes. It does not guarantee the actual resale value of a vehicle. Real prices can vary depending on the car's condition, ownership history, location, and current market demand.

## 👨‍💻 About

This is one of my hands-on projects as I continue learning machine learning and building practical AI applications. My goal is to keep improving my skills by building projects and understanding how each part works.

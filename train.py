import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor


data=pd.read_csv("data/UsedCars_Combined.csv")

x=data[
    ["year",
        "brand",
        "model_name",
        "distance_travelled(kms)",
        "fuel_type",
        "city"]
        ]
y=data["price"]

X_train,X_test,Y_train,Y_test=train_test_split(
    x,y, test_size=0.2,random_state=42
)

print("training row:",len(X_train))
print("test row:",len(X_test))


#separate text columns and numbers colums

cat_cols=["brand","model_name","fuel_type","city"]
num_cols=["year","distance_travelled(kms)"]

#convert the categories into numeric columns
encoder=OneHotEncoder(handle_unknown="ignore")


X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

# Combine numeric columns with encoded text columns
X_train_final = hstack([X_train[num_cols].values, X_train_cat])
X_test_final = hstack([X_test[num_cols].values, X_test_cat])

#linear model
model=LinearRegression()
model.fit(X_train_final,Y_train)

prediction=model.predict(X_test_final)

print("mean absolute error:", mean_absolute_error(Y_test,prediction))
print("R2 score:",r2_score(Y_test,prediction))

print("accutal price:")
print(Y_test.head().to_list())
print("prediction price")
print(prediction[:5])

print(data["price"].describe())
print(data["price"].sort_values().head(10))
print(data["price"].sort_values(ascending=False).head(10))


#random forst model

rf_model=RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_final,Y_train)

rf_predictions=rf_model.predict(X_test_final)

print("MAE:",mean_absolute_error(Y_test,rf_predictions))

print("RANDOM FOREST R2:", r2_score(Y_test,rf_predictions))


#save the encoder
joblib.dump(encoder,"encoder.pkl")
#save linear regression
joblib.dump(model,"liner_model.pkl")
#save random forest
joblib.dump(rf_model,"random_forest_model.pkl")

print("ALL models and encoders saved successfully")
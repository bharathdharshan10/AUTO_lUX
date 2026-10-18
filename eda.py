import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("data/UsedCars_Combined.csv")


plt.figure(figsize=(8,5))
sns.displot(data=data,x="price",bins=30,kde=True)
plt.title("distibution of used cars price")
plt.xlabel("price")
plt.ylabel("numbers of cars")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="distance_travelled(kms)",
    y="price"
)
plt.title("Distance Travelled vs Car Price")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Top 10 car brands by number of listings
top_brands = data["brand"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_brands.index, y=top_brands.values)
plt.title("Top 10 Car Brands in Dataset")
plt.xlabel("Brand")
plt.ylabel("Number of Cars")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fuel type distribution
print(data["fuel_type"].value_counts())

# City distribution
print(data["city"].value_counts())
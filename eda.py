
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("Car details v3.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

# 2. Check missing values and duplicates
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 3. Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)

# 4. Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 5. Convert mileage, engine and max_power into numbers
# Extract numeric values from columns containing units

for col in ["mileage", "engine", "max_power"]:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

# 6. Handle missing values in numerical columns
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 7. Handle missing values in categorical columns
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:", df.shape)

# 8. Check categorical values
print("\nFuel types:")
print(df["fuel"].value_counts())

print("\nTransmission types:")
print(df["transmission"].value_counts())

print("\nOwner types:")
print(df["owner"].value_counts())

# 9. Save the cleaned dataset
df.to_csv("used_cars_cleaned.csv", index=False)
print("\nCleaned dataset saved as used_cars_cleaned.csv")

# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# 10. Summary statistics
print("\nSummary statistics:")
print(df.describe())

print("\nAverage selling price:", df["selling_price"].mean())
print("Median selling price:", df["selling_price"].median())
print("Minimum selling price:", df["selling_price"].min())
print("Maximum selling price:", df["selling_price"].max())

# 11. Distribution of selling prices
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="selling_price", bins=30, kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# 12. Distribution of cars by manufacturing year
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="year", bins=20)
plt.title("Distribution of Cars by Year")
plt.xlabel("Manufacturing Year")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

# 13. Selling price vs kilometres driven
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price")
plt.title("Selling Price vs Kilometres Driven")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price")
plt.tight_layout()
plt.show()

# 14. Average selling price by fuel type
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="fuel", y="selling_price", errorbar=None)
plt.title("Average Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 15. Average selling price by transmission
plt.figure(figsize=(7, 5))
sns.barplot(
    data=df, x="transmission", y="selling_price", errorbar=None
)
plt.title("Average Selling Price by Transmission")
plt.xlabel("Transmission Type")
plt.ylabel("Average Selling Price")
plt.tight_layout()
plt.show()

# 16. Correlation heatmap
num_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(9, 6))
sns.heatmap(
    num_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully.")

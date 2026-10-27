# Used-Car Price Segmentation and Prediction

Exploratory analysis of used-car listing prices, followed by a comparison of four classification models that predict whether a car is **Budget**, **Mid-range** or **Premium** from its listing details.

## Project question
What distinguishes cheap, mid-range and premium used-car listings, and which of four classifiers (SVM, Naive Bayes, Decision Tree, Random Forest) predicts a car's price class best?

## Dataset
- File: `Car_details_v3.csv` (8,128 rows, 13 columns); 6,912 rows remain after cleaning.
- Fields: name, year, selling price, kilometres driven, fuel, seller type, transmission, owner count, mileage, engine, max power, torque, seats.
- Cleaning: removed 1,202 exact duplicate rows (about 15%), 5 "Test Drive Car" rows and 9 rows with implausible mileage; parsed text specs (kmpl, CC, bhp) into numbers.
- Reference year for vehicle age is 2020, the newest model year in the file.
- Two other candidate datasets were reviewed and not used: the earlier `Dataset.csv` (only 1,295 unique rows, few price drivers) and `used_cars.csv` (US market, US dollars).

## Notebook structure
| Part | Content |
|---|---|
| Setup and cleaning | Imports, data validation, duplicate removal, data-quality audit |
| **A. EDA** | Price histogram + KDE, violin plot, count plots, age-vs-mileage scatter, correlation heatmap |
| **B. Five objectives** | 1) brand vs price, 2) vehicle age vs price, 3) mileage vs price, 4) fuel type and transmission vs price, 5) ownership history and seller type vs price |
| **C. Classification** | Target creation, shared preprocessing, 5-fold cross-validated tuning, test-set evaluation, confusion matrices, feature importance, decision-tree plot |
| Conclusions | Results, limitations, next steps |

## Method (Part C)
- **Target:** Budget / Mid-range / Premium, using the 33rd and 67th price percentiles of the **training set only** (≤ ₹300,000 / ≤ ₹550,000 / above). Price is never used as a predictor.
- **Split:** 80% train / 20% test (1,383 test cars), fixed random seed.
- **Features:** age, kilometres driven, fuel efficiency, engine size, max power, seats, brand, model family, fuel, seller type, transmission, owner count.
- **Preprocessing:** median imputation and scaling for numbers; one-hot encoding for categories (rare categories grouped).
- **Tuning:** grid search with 5-fold stratified cross-validation on the training set, scored by macro-F1. The test set is used once, at the end.

## Results (held-out test set)
| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| **SVM (RBF)** | **85.7%** | **0.858** | **0.964** |
| Random Forest | 85.2% | 0.852 | 0.962 |
| Decision Tree | 82.5% | 0.823 | 0.909 |
| Naive Bayes | 73.5% | 0.734 | 0.890 |

- **Best fit: SVM (RBF)**, chosen on cross-validated macro-F1 and confirmed on the test set. Random Forest is within one percentage point, so the two are effectively tied.
- Mistakes are almost always between neighbouring classes; Budget-vs-Premium confusion is only 0.14% for the best model.
- Vehicle age is by far the most important feature, followed by model family, max power and fuel.
- Naive Bayes is weakest because its independence assumption does not hold for correlated features such as age, mileage, engine size and power.

## Key findings
- Luxury brands (BMW, Mercedes-Benz) have median prices several times those of mass-market brands.
- Price falls steadily with age (roughly 13% per year in a simple trend) and more gently with mileage.
- Diesel and automatic cars list higher than petrol and manual cars, partly because they tend to be newer and more powerful.
- Median price falls with each extra previous owner, but later-owner cars are also older, so part of the gap is age.

## Limitations
- Prices are listing prices from a roughly 2020 snapshot, not confirmed sale prices.
- Condition, accident history, service records and location are not in the data.
- Price classes are relative to this dataset; findings are associations, not causal claims.
- Results come from one train/test split plus cross-validation on the training set.

## How to run
1. Put `Car_details_v3.csv` in the same folder as the notebook.
2. Install the dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`
3. Open `Used_Car_Project.ipynb` and run all cells (the tuning step takes a few minutes).

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn.

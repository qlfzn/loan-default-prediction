import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


class LoanDefaultPrediction:
    def __init__(self) -> None:
        print("(" * 10, "LOAN DEFAULT PREDICTION", ")" * 10)
        self.data_path = "./data/Loan_default.csv"

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def explore_data(self, df: pd.DataFrame):
        # things to explore: shape, columns, unique values, datatype, null values
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {df.columns}")
        print(df.describe())
        print(df.isnull())

        # value counts for default vs non-default
        distributions = df["Default"].value_counts(normalize=True)
        percentages = distributions * 100
        print(f"\nDistribution of Default column: {percentages}")

    def create_visualisation(self, df: pd.DataFrame):
        # viz: default vs. non-default, correlation

        # 1st chart: default vs non-default
        plt.figure(figsize=(7, 5))
        sns.countplot(df, x="Default")
        plt.title("Default vs. Non-Default")
        # plt.show()
        plt.savefig("./viz/Target Class Distribution.jpg")

        # 3rd chart: distribution side by side
        features = ["CreditScore", "Income", "LoanAmount", "InterestRate", "DTIRatio"]
        fig, axes = plt.subplots(1, len(features), figsize=(20, 5))

        for i, feature in enumerate(features):
            sns.boxplot(x="Default", y=feature, data=df, ax=axes[i])
            axes[i].set_title(f"{feature} by Default Status")

        plt.tight_layout()
        # plt.show()
        plt.savefig("./viz/Feature by Default.jpg")

        # check mean of features for each class
        avg_df = df.groupby("Default")[features].mean()
        print(avg_df)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        # 4th chart: analyse distribution across all values
        for i, feature in enumerate(features):
            sns.kdeplot(
                data=df[df["Default"] == 0][feature],
                ax=axes[i],
                label="Non-Default",
                fill=True,
                alpha=0.5,
                color="steelblue",
            )
            sns.kdeplot(
                data=df[df["Default"] == 1][feature],
                ax=axes[i],
                label="Default",
                fill=True,
                alpha=0.5,
                color="tomato",
            )
            axes[i].set_title(f"{feature} Distribution")
            axes[i].legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig("./viz/KDEplot.jpg")

    def prepare_data(self, df: pd.DataFrame):
        # encode categorical data
        le = LabelEncoder()
        features = [
            "Education",
            "EmploymentType",
            "MaritalStatus",
            "HasMortgage",
            "HasDependents",
            "LoanPurpose",
            "HasCoSigner",
        ]

        for f in features:
            df[f] = le.fit_transform(df[f])
            print(f"\nAfter encoding: \n {df[f].head(10)}")

        # split train-test
        X = df.drop(["LoanID", "Default"], axis=1)
        y = df["Default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # normalise data using min-max scaling
        mms = MinMaxScaler()
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[numeric_cols] = mms.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = mms.transform(X_test[numeric_cols])

        # oversample data for minority class
        sm = SMOTE(random_state=42)
        print("\nApplying SMOTE for oversampling minority class\n")
        print(f"Before SMOTE: {y_train.value_counts()}")
        X_train, y_train = sm.fit_resample(X=X_train, y=y_train)
        print(f"After SMOTE: {y_train.value_counts()}")

        print(f"Train size: {X_train.shape[0]} samples")
        print(f"Test size: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test):
        # create pipeline
        lr = LogisticRegression()

        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)

        return predictions

    def evaluate_model(self, preds, y_test):
        matrix = confusion_matrix(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        return matrix, accuracy, report


if __name__ == "__main__":
    pipeline = LoanDefaultPrediction()

    # load data
    data = pipeline.load_data()
    print(data.head())

    # data exploration
    pipeline.explore_data(data)
    pipeline.create_visualisation(data)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(data)
    print("\nX_train: \n", X_train[5:], "\nyTrain\n", y_train[5:])
    
    # modeling and evaluation
    predictions = pipeline.train_model(X_train, y_train, X_test)

    matrix, acc_results, class_results = pipeline.evaluate_model(predictions, y_test)
    disp = ConfusionMatrixDisplay(matrix)
    print("\n", disp.plot())
    plt.savefig("./viz/Confusion Matrix")

    print("\nResults: \n Accuracy: ", acc_results, "\n", class_results)
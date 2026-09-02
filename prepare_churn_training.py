import pandas as pd

from src.preprocessing.churn_preprocessing import ChurnPreprocessor

from src.training.data_split import ChurnDataSplitter

def main():
    df = pd.read_csv("artifacts/churn/customer_churn_dataset.csv")

    preprocessor = (ChurnPreprocessor())

    X, y = (preprocessor.prepare_features(df))

    splitter = (ChurnDataSplitter())

    X_train,X_test,y_train,y_test = splitter.split_data(X,y)

    print("\nTraining Feature Shape:")
    print(X_train.shape)
    print("\nTesting Feature Shape:")
    print(X_test.shape)
    print("\nTraining Churn Distribution:")
    print(y_train.value_counts(normalize=True)* 100)
    print("\nTesting Churn Distribution:")
    print(y_test.value_counts(normalize=True)* 100)


if __name__ == "__main__":

    main()
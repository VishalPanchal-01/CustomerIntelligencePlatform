import pandas as pd

from sklearn.pipeline import Pipeline

from src.training.logistic_model import LogisticChurnModel

def test_logistic_churn_model():
    X_train = pd.DataFrame({
            "Recency": [10,20,30,100,120,150],
            "Frequency": [10,8,6,2,1,1],
            "Monetary": [1000,900,700,200,100,50],
            "TotalItems": [100,90,70,20,10,5],
            "AverageOrderValue": [100,112.5,116.67,100,100,50],
            "Tenure": [300,250,200,50,30,10]
        })

    y_train = pd.Series([0,0,0,1,1,1])

    trainer = LogisticChurnModel()

    model = trainer.train(X_train,y_train)

    assert model is not None

    assert isinstance(model,Pipeline)

    predictions = model.predict(X_train)

    assert len(predictions) == 6

    assert set(predictions).issubset({0, 1})
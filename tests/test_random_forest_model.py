import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.training.random_forest_model import RandomForestChurnModel

def test_random_forest_churn_model():
    X_train = pd.DataFrame({
            "Recency": [10,20,30,40,100,120,150,180],
            "Frequency": [10,9,7,6,3,2,1,1],
            "Monetary": [1500,1200,1000,800,400,250,100,50],
            "TotalItems": [150,120,100,80,40,25,10,5],
            "AverageOrderValue": [150,133,143,133,133,125,100,50],
            "Tenure": [350,320,280,250,100,70,30,10]
        })

    y_train = pd.Series([0,0,0,0,1,1,1,1])

    trainer = (RandomForestChurnModel())

    model = trainer.train(X_train,y_train)

    assert model is not None

    assert isinstance(model,RandomForestClassifier)

    predictions = model.predict(X_train)

    assert len(predictions) == 8

    assert set(predictions).issubset({0, 1})
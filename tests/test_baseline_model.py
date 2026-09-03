import pandas as pd

from src.training.baseline_model import BaselineChurnModel

def test_baseline_model():
    X_train = pd.DataFrame({
            "Recency": [10,20,30,40,50],
            "Frequency": [5,4,3,2,1]
        })

    y_train = pd.Series([1,1,1,0,0])

    trainer = BaselineChurnModel()

    model = trainer.train(X_train,y_train)

    predictions = model.predict(X_train)

    assert model is not None

    assert len(predictions) == 5

    assert set(predictions) == {1}
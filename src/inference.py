import numpy as np
import pandas as pd

class Inference:
    def __init__(self , scaler):
        self.scaler = scaler


    def predict(self , X , model):
        if isinstance(X , (float , int)):
            X = np.array([X]).reshape(1,-1)

        results = model.predict(X)

        return self.get_results(results)





    def get_results(self, preds):
        df_out = pd.DataFrame(self.scaler.inverse_transform(preds)
                                ,columns=self.scaler.get_feature_names_out())  /3600

        return df_out
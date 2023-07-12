import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

from utils.evaluation import eval_model


class ModelTraining:
    def __init__(self , model , small = False):
        self.model = model
        self.small = small


    def train_df(self , trans_df ):
        
        X = trans_df[["NewCFactor"]]
        
        y = trans_df[["RunLength_Cum", "N_Pulses_Cum"]]
        
        X_train , X_test ,y_train , y_test = train_test_split(X , y ,test_size = 0.2)
        
        ##### Scaling
        
        self.scaler = StandardScaler()
        
        y_train = pd.DataFrame(self.scaler.fit_transform(y_train) 
                                , columns=["RunLength_Cum","N_Pulses_Cum"])
        
        y_test = pd.DataFrame(self.scaler.transform(y_test)
                                , columns=["RunLength_Cum","N_Pulses_Cum"])

        
        

        
        
        
        self.model.fit(X_train ,y_train)
        
        
        train_pred = self.model.predict(X_train)

        eval_model(X_train, y_train ,train_pred , small = self.small)
        
        
        test_pred = self.model.predict(X_test)
        
        eval_model(X_test , y_test ,test_pred,test=True  ,small = self.small)
        
        
        return self
    
    def predict(self , X):
        if isinstance(X , (float , int)):
            X = np.array([X]).reshape(1,-1)

        results = self.model.predict(X)

        return self.get_results(results)





    def get_results(self, preds):
        df_out = pd.DataFrame(self.scaler.inverse_transform(preds)
                                ,columns=self.scaler.get_feature_names_out())  /3600

        return df_out


if __name__ == '__main__':
    fail_point = 0.9053574604

    # lr =train_df(df_large)
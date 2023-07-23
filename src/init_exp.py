import numpy as np
import pandas as pd

#["RunLength_Cum" , "N_Pulses_Cum"]

class ExpRegression:
    def __init__(self):
        self.coefs_run = None
        self.coefs_pulses = None


    def fit(self , x , y):
        self.coefs_run = np.polyfit(x.reshape(-1), y["RunLength_Cum"], 1)
        self.coefs_pulses = np.polyfit(x.reshape(-1), y["N_Pulses_Cum"], 1)


        return self.coefs_run , self.coefs_pulses


    def predict(self ,  x ):
        intercept , slope = self.coefs
        preds = np.exp(intercept) * np.exp(slope * x)
        return preds


# def init_exp():
      
#       model_large_init = MultiOutputRegressor(LinearRegression())
#       model_small_init = MultiOutputRegressor(LinearRegression())

#       return model_large_init , model_small_init

if __name__ == "__main__":
    df = pd.read_csv("data/df_large_prep.csv",index_col=0)
    
    exp_reg = ExpRegression()



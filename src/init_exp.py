import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#["RunLength_Cum" , "N_Pulses_Cum"]

class ExpRegression:
    def __init__(self):
        self.coefs_run = None
        self.coefs_pulses = None



    def fit(self , x , y):
        # return x
        self.coefs_run = np.polyfit(x["NewCFactor"].values.reshape(-1), y["RunLength_Cum"], 1)
        self.coefs_pulses = np.polyfit(x["NewCFactor"].values.reshape(-1), y["N_Pulses_Cum"], 1)
        self.plot_train( x , y)

        return self.coefs_run , self.coefs_pulses

    

    def plot_train(self,  x , y):
         plt.title()
         plt.plot(x , y)
         plt.plot(x , x * self.coefs_run[0] + self.coefs_run[1])
         plt.plot(x , x * self.coefs_pulses[0] + self.coefs_pulses[1])
         plt.show()


    def predict(self ,  x ):
        intercept_run , slope_run = self.coefs_run
        preds_run = np.exp(intercept_run) * np.exp(slope_run * x)

        intercept_pulses , slope_pulses = self.coefs_pulses
        preds_pulses = np.exp(intercept_pulses) * np.exp(slope_pulses * x)

        return preds_run , preds_pulses



def init_exp():
      
      model_large_init = ExpRegression()
      model_small_init = ExpRegression()
      return model_large_init , model_small_init

if __name__ == "__main__":
    df = pd.read_csv("data/df_large_prep.csv",index_col=0)
    
    exp_reg = ExpRegression()

    large_model , small_model = init_exp()
    large_model.fit(df[["NewCFactor"]] , df[["RunLength_Cum", "N_Pulses_Cum"]] )



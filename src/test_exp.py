import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 
class ExponentialModel:
    def __init__(self):
        pass

    def exponential_func(self, x, a, b):
        return a * np.exp(b * x)

    def fit(self, x_data, y_data): 
        x_data = x_data#.reshape(-1,) 
        y_data = y_data#.reshape(-1,) 
        params, _ = curve_fit(self.exponential_func, x_data, y_data, maxfev=50000)
        self.a_fit, self.b_fit = params
        self.x_fit = np.linspace(min(x_data), max(x_data), len(x_data))
        self.y_fit = self.exponential_func(self.x_fit, self.a_fit, self.b_fit)
        return x_data, y_data, x_fit, y_fit, a_fit, b_fit

    def predict(self, x):
        return self.exponential_func(x, self.a_fit, self.b_fit)

def plot_and_predict(model, x_data, y_data, title):
    model.fit(x_data, y_data)

    plt.scatter(x_data, y_data, label='Actual Data')
    plt.plot(model.x_fit, model.y_fit, color='red', label='Fitted Curve')
    plt.title(title)
    plt.xlabel('NewCFactor')
    plt.ylabel('Values')  # Replace with the appropriate label
    plt.legend()
    plt.show()

    # Example prediction for a new x value
    new_x_value = 10.0  # Replace with the desired NewCFactor value
    predicted_value = model.predict(new_x_value)
    print(f"Prediction for NewCFactor {new_x_value}: {predicted_value}")





if __name__ == "__main__":
    df_large = pd.read_csv("data/df_large_prep.csv",index_col=0)
    df_small = df_large.copy()
        
    # Assuming df_large and df_small are your DataFrames with the required columns
    # Replace 'N_Pulses_Cum' and 'RunLength_Cum' with the actual column names
    model_large = ExponentialModel()
    plot_and_predict(model_large, df_large['NewCFactor'], df_large['N_Pulses_Cum'], 'Large Model - N_Pulses_Cum')
    plot_and_predict(model_large, df_large['NewCFactor'], df_large['RunLength_Cum'], 'Large Model - RunLength_Cum')

    model_small = ExponentialModel()
    plot_and_predict(model_small, df_small['NewCFactor'], df_small['N_Pulses_Cum'], 'Small Model - N_Pulses_Cum')
    plot_and_predict(model_small, df_small['NewCFactor'], df_small['RunLength_Cum'], 'Small Model - RunLength_Cum')

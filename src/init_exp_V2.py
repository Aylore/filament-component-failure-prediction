import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from utils.evaluation import eval_model , calc_rmse


from utils.preprocess import DataProcessor


def plot_pred( x_data, y_data, x_fit, y_fit, a_fit, b_fit ,  x_fit2 , y_fit2,  idx,  target_name   , df):
    targets = ["N_Pulses_Cum" , "RunLength_Cum"]
    x_scaled , y_scaled , _ = scale(df , targets[0])
    x_scaled2 , y_scaled2 , _ = scale(df , targets[1])
    if idx ==0:
        # plt.scatter(x_data, y_data, color='gray', label='Data for Large')
        plt.scatter(x_scaled , y_scaled , color='cyan' , label = f'Actual values for large {targets[0]}')
        plt.scatter(x_scaled2 , y_scaled2 , color='blue' , label = f'Actual values for large {targets[1]}')

    else:
        plt.scatter(x_scaled , y_scaled , color='blue' , label = f'Actual values for small {targets[0]}')
        plt.scatter(x_scaled2 , y_scaled2 , color='cyan' , label = f'Actual values for small {targets[1]}')  
    plt.plot(x_fit, y_fit, color='grey', label=f'Exponential Regression {target_name[0]} predictions' )
    plt.plot(x_fit2, y_fit2, color='red', label=f'Exponential Regression {target_name[1]} predictions')


    plt.xlabel('NewCFactor')
    plt.ylabel("Cumulative Runlength & N_pulses")
    # plt.title(f'Exponential Regression of {self.data_type}')
    plt.legend()
    plt.show()
    


    # print(f"Fitted a: {a_fit}")
    # print(f"Fitted b: {b_fit}")


def exponential_func( x, a, b):
    return a * np.exp(b * x)

def fit( x_data, y_data , target_name='N_Pulses_Cum' , idx =0): 
    x_data = x_data.reshape(-1,) 
    y_data = y_data.reshape(-1,) 
    params, _ = curve_fit(exponential_func, x_data, y_data, maxfev=50000)
    a_fit, b_fit = params
    x_fit = np.linspace(min(x_data), max(x_data), len(x_data))
    y_fit = exponential_func(x_fit, a_fit, b_fit)
    ll = ['large_N_Pulses_Cum' , 'large_RunLength_Cum' , 'small_N_Pulses_Cum' , 'small_RunLength_Cum']
    print(f"rmse {'large' if idx ==0  else 'small'}_{target_name} : {  calc_rmse(y_data ,y_fit )}")
    return x_data, y_data, x_fit, y_fit, a_fit, b_fit


def predict(value, a_fit , b_fit):
    return exponential_func(value , a_fit , b_fit)

def scale(df , target_name):
    scaler_X = MinMaxScaler(feature_range=(1, 10))
    scaler_y = MinMaxScaler(feature_range=(1, 10))
    x_scaled = scaler_X.fit_transform(df.NewCFactor.values.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(df[target_name].values.reshape(-1, 1))
    return x_scaled , y_scaled , scaler_X 





def do_all(df_large , df_small):
    models = {}
    target_names = ['N_Pulses_Cum' , 'RunLength_Cum']
    dfs = [df_large , df_small]
    for idx , df in enumerate(dfs):
        for target_name in target_names:
            if idx ==0:
                df_name = "large"
            else:
                df_name = "small"
            x_scaled , y_scaled , x_scaler = scale(df , target_name)
            params = fit(x_scaled , y_scaled , target_name , idx) 
            models[df_name + "_" + str(target_name) ] = params

        ## plot it as multioutput model
        keys_list = ["x_fit" , "y_fit"]
        if idx ==0:
            x_fit2 = models[f"large_{target_names[1]}"][2]
            y_fit2 = models[f"large_{target_names[1]}"][3]
            plot_pred(*models[f"large_{target_names[0]}"] ,x_fit2 , y_fit2 ,idx , target_names ,df)
            # plot_pred(*models[f"large_{target_names[1]}"] , target_names[1])

        else:
            x_fit2 = models[f"small_{target_names[1]}"][2]
            y_fit2 = models[f"small_{target_names[1]}"][3]
            plot_pred(*models[f"small_{target_names[0]}"] , x_fit2,y_fit2  ,idx , target_names , df)
            # plot_pred(*models[f"small_{target_names[1]}"] , target_names[1])

    return models , x_scaler


def predict_all(models , value_to_predict , x_scaler):
    # large_pulses , large_run , small_pulses , small_run = models.values() 
    scaled_value = x_scaler.transform(np.array(value_to_predict).reshape(1 , -1))
    for i in models.keys():
        res =  predict(scaled_value , models[i][-2] , models[i][-1] )
        print(f"{i} prediciotn : {res}")



def do_exp(df_large , df_small):
      fail_point = 0.9453574604

      models  , x_scaler= do_all(df_large , df_small)
      predict_all(models , fail_point , x_scaler)




if __name__ == "__main__":
    # df_large = pd.read_csv("data/df_large_prep.csv", index_col=0)
    # df_small = pd.read_csv("data/df_small_prep.csv", index_col=0)
        ## Read Data
    df = pd.read_xml("data/KUH.XML")

    ## Preprocess Data
    ## Initialize Date preprocessor
    preprocess = DataProcessor()
    df_large , df_small = preprocess.prep_data(df)
    target_name = ['N_Pulses_Cum' , 'RunLength_Cum']
    ## create pipeline
    ### first scaled
    # x_scaled , y_scaled , scaler_X = scale(df_large , target_name[0])
    # ### Fit 
    # params = fit(x_scaled , y_scaled)
    # ### plot
    # plot_pred(*params, target_name[0])

    # value_to_predict = 2

    # predict(value_to_predict , params[-2] , params[-1])

    fail_point = 0.9453574604

    models  , x_scaler= do_all()
    predict_all(models , fail_point , x_scaler)


    # print(f"Prediction for {value_to_predict}:\nLarge Model: {predictions[0]}\nSmall Model: {predictions[1]}")

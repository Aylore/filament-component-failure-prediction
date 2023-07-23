import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import os

def calc_rmse(y_true , y_preds):

    return mse(y_true , y_preds , squared=False)


def eval_model(X , y, preds , test = False  , small = False):
    if not os.path.exists("plots/"):
        os.mkdir("plots")

    data_type = "train" if not test else "test"
    dataset_type = "Large" if not small else "Small"

    preds = pd.DataFrame(preds , columns= [["RunLength_Cum" , "N_Pulses_Cum"]])
    # print(preds)

    print(f"RMSE for {data_type}_{dataset_type} RunLength_Cum : {calc_rmse(y['RunLength_Cum'] , preds['RunLength_Cum'])}")
    print(f"RMSE for {data_type}_{dataset_type} N_Pulses_Cum : {calc_rmse(y['N_Pulses_Cum'] , preds['N_Pulses_Cum'])}")

    plt.scatter(X , y["RunLength_Cum"] ,label="Actual Values For RunLength_Cum")
    plt.scatter(X , y["N_Pulses_Cum"] ,label="Actual Values For N_pulses_Cum" , color='cyan')

    color_palette = plt.cm.get_cmap('Set1', 2)

    for i ,col in enumerate(preds.columns):
        plt.plot(X   ,preds[col] , label=f"{col}_Predictions" , color=color_palette(i))
    
    plot_title = f"Model Performence On {data_type}_{dataset_type} data"
    plt.title(plot_title)
    plt.legend()

    plt.xlabel("NewCFactor")
    plt.ylabel("Cumulative Runlength & N_pulses")
    plt.savefig(f"plots/{plot_title}_{str(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))}")
    plt.show()

 
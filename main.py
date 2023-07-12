import pandas as pd

from src.train_linear import ModelTraining
from utils.preprocess import DataProcessor


from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

from src.inference import Inference

def main(df_path):

    ## Initialize models
    train_model_large = ModelTraining(model_large_init)
    train_model_small = ModelTraining(model_small_init , small=True)

    ## Read Data
    df = pd.read_xml(df_path)

    ## Preprocess Data
    df_large , df_small = preprocess.prep_data(df)

    ## Train model
    model_large = train_model_large.train_df(df_large )
    model_small = train_model_small.train_df(df_small )

    return model_large , model_small

if __name__ == "__main__":
    path = 'data/MCC.XML'
    
    ## Initialize model
    model_large_init = MultiOutputRegressor(LinearRegression())
    model_small_init = MultiOutputRegressor(LinearRegression())

    ## Initialize Date preprocessor
    preprocess = DataProcessor()
    
    model_large , model_small = main(path)



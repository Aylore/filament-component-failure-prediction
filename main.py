import pandas as pd

from src.train import ModelTraining
from src.init_lr import init_lr
from src.init_exp import ExpRegression

from utils.preprocess import DataProcessor



def main(df_path):

    ## Initialize models
    train_model_large = ModelTraining(model_large_init)
    train_model_small = ModelTraining(model_small_init , small=True)

    ## Read Data
    df = pd.read_xml(df_path)

    ## Preprocess Data
    df_large , df_small = preprocess.prep_data(df)

    #### Temproray for DEV only 
    # # df_large.to_csv("data/df_large_prep.csv")
    # # df_small.to_csv("data/df_small_prep.csv")
    # df_large = pd.read_csv("data/df_large_prep.csv",index_col=0)
    # df_small = pd.read_csv("data/df_small_prep.csv",index_col=0)

    ## Train model
    model_large = train_model_large.train_df(df_large )
    model_small = train_model_small.train_df(df_small )

    return model_large , model_small

if __name__ == "__main__":
    path = 'data/KUH.XML'
    
    ## Initialize model
    model_large_init , model_small_init = init_lr()

    ## Initialize Date preprocessor
    preprocess = DataProcessor()
    
    model_large , model_small = main(path)




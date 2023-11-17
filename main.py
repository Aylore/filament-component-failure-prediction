import pandas as pd

from src.train import ModelTraining
from src.init_lr import init_lr
from src.init_rf import init_rf


from utils.preprocess import DataProcessor



def main(df_path):

    ## Initialize models
    train_model_large = ModelTraining(model_large_init )
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
    fail_point = 0.9453574604

    path = 'data/KUH.XML'
    
    ## Initialize model     
    model_large_init , model_small_init = init_rf()

    ## Initialize Date preprocessor
    preprocess = DataProcessor()
    
    model_large , model_small = main(path)

    
    print(f"model_large preds : \n{model_large.predict(fail_point)}\nmodel_small preds : \n{model_small.predict(fail_point)}")



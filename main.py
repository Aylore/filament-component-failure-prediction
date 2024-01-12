import pandas as pd
from src.init_exp_V2 import do_exp

from src.train import ModelTraining
from src.init_lr import init_lr

from utils.preprocess import DataProcessor



def main(df_path ):
    ## Read Data
    df = pd.read_xml(df_path)

    ## Preprocess Data
    df_large , df_small = preprocess.prep_data(df)

    
    ## Initialize models
    train_model_large = ModelTraining(model_large_init )
    train_model_small = ModelTraining(model_small_init , small=True)

    


    ## Train model
    model_large = train_model_large.train_df(df_large )
    model_small = train_model_small.train_df(df_small )

    ## Exp regression 
    # models  , x_scaler= do_all()
    # predict_all(models , fail_point , x_scaler)

    return model_large , model_small

  


def exp(df_path):
        ## Read Data
    df = pd.read_xml(df_path)

    ## Preprocess Data
    df_large , df_small = preprocess.prep_data(df)

    do_exp(df_large , df_small)




if __name__ == "__main__":
    fail_point = 0.9453574604

    path = 'data/KUH.XML'
    
    use_exp = False         ### change to True to use exponential regerssion 

    if use_exp:
         exp(path)
    else:
        ## Initialize model     
        model_large_init , model_small_init = init_lr() 

        ## Initialize Date preprocessor
        preprocess = DataProcessor()
        
        model_large , model_small = main(path )

    
        print(f"model_large preds : \n{model_large.predict(fail_point)}\nmodel_small preds : \n{model_small.predict(fail_point)}")



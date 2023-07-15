from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression




def init_lr():

    model_large_init = MultiOutputRegressor(LinearRegression())
    model_small_init = MultiOutputRegressor(LinearRegression())

    return model_large_init , model_small_init
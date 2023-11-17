from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor



def init_rf():

    model_large_init = MultiOutputRegressor(RandomForestRegressor(random_state=3407))
    model_small_init = MultiOutputRegressor(RandomForestRegressor(random_state=3407))

    return model_large_init , model_small_init
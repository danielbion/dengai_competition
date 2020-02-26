import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split

global_features1 = [
    'ndvi_ne',
    'ndvi_nw',
    'ndvi_se',
    'ndvi_sw',
    #'precipitation_amt_mm',
    #'reanalysis_air_temp_k',
    'reanalysis_avg_temp_k',
    #'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k', 
    'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent',
    'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg',
    'reanalysis_tdtr_k',
    #'station_avg_temp_c',
    #'station_diur_temp_rng_c', 
    #'station_max_temp_c',
    #'station_min_temp_c',
    #'station_precip_mm'
]

global_features = [
    'ndvi_ne',
    'ndvi_nw',
    'ndvi_se',
    'ndvi_sw',
    'precipitation_amt_mm',
    'reanalysis_air_temp_k',
    'reanalysis_avg_temp_k',
    'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k', 
    'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent',
    'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg',
    'reanalysis_tdtr_k',
    'station_avg_temp_c',
    'station_diur_temp_rng_c', 
    'station_max_temp_c',
    'station_min_temp_c',
    'station_precip_mm',
    'reanalysis_sat_precip_amt_mm_window_2',
    'reanalysis_sat_precip_amt_mm_window_4',
    'reanalysis_avg_temp_k_window_2', 
    'reanalysis_avg_temp_k_window_4',
    'reanalysis_specific_humidity_g_per_kg_window_2',
    'reanalysis_specific_humidity_g_per_kg_window_4'
]

global_params = {
    'iterations': [50, 100, 300],
    'learning_rate': [0.05, 0.1],
    'depth': [3, 5, 7],
    'l2_leaf_reg': [1, 5, 9]
}

model_params = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 3
}

    
def train_catboost(train_pool, eval_pool, grid_search, grid_params):    
    model = CatBoostRegressor()
    
    if(grid_search):
        grid_search_result = model.grid_search(
            param_grid = grid_params, 
            X = train_pool,
            cv = 3,
            verbose = 10)

        p = pd.DataFrame.from_dict(grid_search_result).params
        print(p)

        model = CatBoostRegressor(iterations = p.iterations,
                                  learning_rate = p.learning_rate,
                                  depth = p.depth,
                                  l2_leaf_reg = p.l2_leaf_reg)
    else:
        model = CatBoostRegressor(iterations = model_params['iterations'],
                              learning_rate = model_params['learning_rate'],
                              depth = model_params['depth'], has_time = True)
    
    model.fit(train_pool, eval_set=eval_pool, verbose = 100)
    
    return model

def get_metrics(model, train_pool, eval_pool):
    train_metrics = model.eval_metrics(train_pool, ['RMSE'], ntree_start = model.tree_count_ - 1)
    test_metrics = model.eval_metrics(eval_pool, ['RMSE'], ntree_start = model.tree_count_ - 1)

    metrics = pd.DataFrame.from_dict(train_metrics)\
    .append(pd.DataFrame.from_dict(test_metrics), ignore_index=True)\
    .transpose().rename(columns={0:'Train', 1:'Test'})
    return metrics

def split_data_city(city, train_features, train_labels, cat_features, features, remove_outliers):
    city_train_features = train_features[train_features.city == city].reset_index(drop = True).copy()
    city_train_labels = train_labels[train_labels.city == city].reset_index(drop = True).copy()

    if(remove_outliers):
        upper_bound = city_train_labels.total_cases.mean() + (city_train_labels.total_cases.std() * 3)
        without_outliers = city_train_labels.total_cases < upper_bound

        city_train_features = city_train_features[without_outliers]
        city_train_labels = city_train_labels[without_outliers]
        
    splitted_data = train_test_split(city_train_features[features], city_train_labels.total_cases, test_size=0.4, shuffle = False)
    train_data, eval_data, train_target, eval_target = splitted_data

    train_pool = Pool(data = train_data, label = train_target, cat_features = cat_features)
    eval_pool = Pool(data = eval_data, label = eval_target, cat_features = cat_features)
    return (train_pool, eval_pool)

def train_city(city, train_features, train_labels, features, cat_features, grid_params, remove_outliers = False, grid_search = True):
    print(city)
    train_pool, eval_pool = split_data_city(city, train_features, train_labels, cat_features, features[city], remove_outliers)

    model = train_catboost(train_pool, eval_pool, grid_search, grid_params[city])
    model.save_model(city)

    metrics = get_metrics(model, train_pool, eval_pool)

    print(metrics)
    print(model.get_feature_importance(prettified = True))
    return model

def train_model(train_features, train_labels):
    splitted_data = train_test_split(train_features[global_features], train_labels.total_cases, test_size=0.4, shuffle = False)
    train_data, eval_data, train_target, eval_target = splitted_data

    train_pool = Pool(data = train_data, label = train_target)
    eval_pool = Pool(data = eval_data, label = eval_target)

    model = train_catboost(train_pool, eval_pool, True, global_params)
    model.save_model('global')

    metrics = get_metrics(model, train_pool, eval_pool)

    print(metrics)
    print(model.get_feature_importance(prettified = True))
    return model

def predict_model(test_features, pks):
    model = CatBoostRegressor()
    model.load_model('global')
    test_pred = model.predict(test_features[global_features])
    test_pred = pd.DataFrame(test_pred).rename(columns = {0:'total_cases'}).total_cases.map(round)
    result = pd.concat([test_features[pks], test_pred], axis = 1)
    return result

def predict_city(city, test_features, features, pks):
    city_test_features = test_features[test_features.city == city].reset_index(drop = True).copy()
    model = CatBoostRegressor()
    model.load_model(city)
    test_pred = model.predict(city_test_features[features[city]])
    test_pred = pd.DataFrame(test_pred).rename(columns = {0:'total_cases'}).total_cases.map(round)
    result = pd.concat([city_test_features[pks], test_pred], axis = 1)
    return result
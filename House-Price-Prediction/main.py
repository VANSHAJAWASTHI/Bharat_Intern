import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.preprocessing import StandardScaler as ss, OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer as ct
from sklearn.pipeline import Pipeline as pp
from sklearn.impute import SimpleImputer as si
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2
import tensorflow as tf
import json

def ld(fp):
    d = pd.read_csv(fp)
    print("Columns in the CSV file:", d.columns)
    y = d.pop('price')  # Assuming 'price' is the target variable
    xt, xts, yt, yts = tts(d, y, test_size=0.2, random_state=42)
    return xt, xts, yt, yts

def ppd(xt, xts):
    nf = xt.select_dtypes(include=['int64', 'float64']).columns
    cf = xt.select_dtypes(include=['object']).columns
    
    # Pipeline for numerical features
    np = pp([('imp', si(strategy='median')), ('scl', ss())])
    
    # Pipeline for categorical features
    cp = pp([('imp', si(strategy='most_frequent')), ('enc', ohe(handle_unknown='ignore'))])
    
    # ColumnTransformer for both numerical and categorical features
    ppct = ct([('num', np, nf), ('cat', cp, cf)])
    
    # Fit and transform training data
    xt = ppct.fit_transform(xt)
    
    # Transform testing data
    xts = ppct.transform(xts)
    
    return xt, xts, ppct

def trf(xt, yt):
    pg = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    m = gscv(rfr(random_state=42), pg, cv=3, scoring='neg_mean_squared_error')
    m.fit(xt, yt)
    joblib.dump(m.best_estimator_, 'output/rf.pkl')
    return m.best_estimator_

def ttf(xt, yt):
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[xt.shape[1]]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    m.fit(xt, yt, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    m.save('output/tf.h5')
    return m

def ev(rfm, tfm, ppct, xts, yts):
    rfp = rfm.predict(xts)
    tfp = tfm.predict(xts).flatten()
    m = {
        'rf': {'mse': mse(yts, rfp), 'mae': mae(yts, rfp), 'r2': r2(yts, rfp)},
        'tf': {'mse': mse(yts, tfp), 'mae': mae(yts, tfp), 'r2': r2(yts, tfp)}
    }
    return m

def sr(m):
    with open('output/results.txt', 'w') as f:
        f.write(f"RF MSE: {m['rf']['mse']}, MAE: {m['rf']['mae']}, R²: {m['rf']['r2']}\n")
        f.write(f"TF MSE: {m['tf']['mse']}, MAE: {m['tf']['mae']}, R²: {m['tf']['r2']}\n")
    with open('output/metrics.json', 'w') as f:
        json.dump(m, f, indent=4)

def main():
    data_file = 'Housing.csv'
    xt, xts, yt, yts = ld(data_file)
    xt, xts, ppct = ppd(xt, xts)
    rfm = trf(xt, yt)
    tfm = ttf(xt, yt)
    m = ev(rfm, tfm, ppct, xts, yts)
    sr(m)

if __name__ == '__main__':
    main()

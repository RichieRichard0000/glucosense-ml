import numpy as np
import pandas as pd
import pickle
import feature_eng
from sklearn.base import BaseEstimator, TransformerMixin

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

def perform_diabetes_test(features):
    model = load_model('models/stack.pkl')
    test_label = model.predict(features)
    if test_label == 0:
        return "Non-diabetic"
    elif test_label == 1:
        return "Pre-diabetic"
    else:
        return "Highly diabetic"

def perform_feature_selection(test_data):
    selector1 = load_model("models/selector.pkl")
    selector2 = load_model("models/corr_scaler.pkl")
    normalizer = load_model("models/normalizer.pkl")

    selected_names = test_data.columns[selector1.get_support()]
    features = pd.DataFrame(data=selector1.transform(test_data), columns=selected_names)
    
    X_new = selector2.transform(features)
    return normalizer.transform(X_new)

def perform_bgl_test(features):
    bgl_model = load_model(file_path='models/AdaBoost.pkl')
    bgl_value = bgl_model.predict(features)[0]
    return np.round(bgl_value, 2)

def remove_irrelevant_data(df):
    # Read the CSV file into a DataFrame, skipping the first 3 rows and setting the 4th row as header
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    return df.reset_index(drop=True)

def generate_data(sensors_data, body_vitals):
    cleaned_df = remove_irrelevant_data(sensors_data)
    features_df = feature_eng.generate_features(df=cleaned_df)
    final_df = pd.concat([body_vitals, features_df], axis=1)
    return final_df

class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Compute the Spearman correlation matrix
        corr_matrix = X.corr(method='spearman').abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify correlated feature pairs
        correlated_pairs = [(col, row) for col in upper.columns for row in upper.index if upper.at[row, col] >= self.threshold]
        
        # Determine which feature to drop based on variance
        drop_set = set()
        for col, row in correlated_pairs:
            if X[col].var() >= X[row].var():
                drop_set.add(row)  # Drop the feature with lower variance
            else:
                drop_set.add(col)
        
        self.to_drop_ = list(drop_set)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.to_drop_, errors='ignore')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


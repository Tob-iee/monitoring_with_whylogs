import os
import sys

import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def read_prep_data (filename: str):
    df = pd.read_csv(filename)

    # features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    dicts = df.to_dict(orient='records')
    return dicts


def load_apply_model(features, model_file):

    with open(model_file, "rb") as f_in:
        (dv, model) = pickle.load(f_in)

    X = dv.transform(features)
    preds = model.predict(X)
    return preds


def run(filename, model_file):
    # filename = sys.argv[1]
    # model_file = sys.argv[2]

    features = read_prep_data (filename)
    output = load_apply_model(features, model_file)
    return output
    # print(list(output).count(0))

# if __name__ == '__main__':
#     run()
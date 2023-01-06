# master script for the preprocessing of water pump data
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from heatmap_pearson import *
from KNN_imputation import *


# perform preprocessing in 5 steps.
# 1: basic handling
# 2: call heatmap (accepts threshold),
# reduce continuous-variable columns from there
# 3: KNN imputation (accepts number of nearest neighbors)
# 4: final preprocessing
# 5: test-train split and save files
def main():
    # 1. Basic handling
    parser = argparse.ArgumentParser()
    parser.add_argument('x', help='features of data')
    parser.add_argument('y', help='labels of data')
    parser.add_argument('threshold', help='Pearson heatmap threshold')
    parser.add_argument('k', help='number of neighbors for KNN imputation')
    args = parser.parse_args()
    x = str(args.x)
    y = str(args.y)
    threshold = float(args.threshold)
    k = int(args.k)
    df_x = pd.DataFrame(pd.read_csv(x)).fillna(np.nan)  # dummy text to identify later
    # replace numerical zeros in tsh, gps height construction year with np.nan
    to_replace = {'amount_tsh': {0: np.nan},
                  'gps_height': {0: np.nan},
                  'construction_year': {0: np.nan}}
    df_x = df_x.replace(to_replace)
    df_y = pd.DataFrame(pd.read_csv(y))
    y = list(df_y['status_group'])
    df_y = pd.DataFrame(np.c_[y], columns=['label'])

    # 2: call the Pearson Heatmap, return the outputted Dataframe for x,
    # where it is called reduced because unimportant numerical columns
    # are removed from the dataframe. See heatmap_pearson.py

    # NOTE: some numerical features (eg, construction_year) don't appear
    # in this heatmap. Not sure why, to be debugged. It also might be a good
    # idea later on to do a heatmap with encoded categorical features
    # after imputing. TODO: decide about this
    reduced_x = heatmap(df_x, y, threshold)

    # 3: call the KNNimputer, which returns an imputed Dataframe for x
    imputed_x = KNNimputation(reduced_x, k)

    # 4: do final preprocessing
    # Standardization/min-max scaling?
    pass # for now, not needed? # TODO: decide

    # 5: perform test-train split and save csv's
    xTrain, xTest, yTrain, yTest = train_test_split(imputed_x, df_y,
                                                    test_size=0.3,
                                                    random_state=334)
    pd.DataFrame(xTrain, columns=xTrain.columns).to_csv('xTrain.csv', index=False)
    pd.DataFrame(yTrain, columns=yTrain.columns).to_csv('yTrain.csv', index=False)
    pd.DataFrame(xTest, columns=xTest.columns).to_csv('xTest.csv', index=False)
    pd.DataFrame(yTest, columns=yTest.columns).to_csv('yTest.csv', index=False)


if __name__ == '__main__':
    main()

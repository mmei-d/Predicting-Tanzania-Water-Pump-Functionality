import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import KNN  # https://pypi.org/project/fancyimpute/\


def KNNimputation(x, k_param):
    # first, encode the categorical features of x
    # (KNN impute only works for numerical values :( , apparently)
    ord_enc = OrdinalEncoder()

    # latitude and longitude don't need to be imputed or rounded to integers
    x_no_lat_long = x.drop(['latitude', 'longitude'], axis=1)
    x_ord_feats = list(x.columns)
    x_ord_feats.remove('latitude')
    x_ord_feats.remove('longitude')
    # encode the categorical variables
    np_x_ord = ord_enc.fit_transform(x_no_lat_long)

    # new strategy: Applying KNN impute directly
    # to the dataframe takes too much time and memory (13.1 GB estimate),
    # so instead, I will apply KNNimpute in "rolling windows" of a more
    # manageable size. Since features are encoded, I will round to the
    # nearest integer in order for decoding to be possible later.

    # initialize array to copy values to later
    x_imputed = np.empty([np.shape(np_x_ord)[0], np.shape(np_x_ord)[1]])
    window_list = list(range(0, np.shape(np_x_ord)[0], 1000))  # window size is 1000 for now

    # apply KNN-impute within each window, copy imputed filled array to x_imputed
    for w_index in range(len(window_list) - 2):
        cur_x = np.array(np_x_ord[window_list[w_index]:window_list[w_index + 1], :])
        cur_x_filled = KNN(k=k_param).fit_transform(cur_x)
        x_imputed[window_list[w_index]:window_list[w_index + 1], :] = cur_x_filled

    # round everything to the nearest integer (to match with encoding)
    for row in range(np.shape(x_imputed)[0]):
        for col in range(np.shape(x_imputed)[1]):
            x_imputed[row, col] = int(round(x_imputed[row, col], 0))

    # decode, put back as pd DataFrame with latitude and longitude re-included
    x_filled_np = ord_enc.inverse_transform(x_imputed)
    x_filled_df = pd.DataFrame(x_filled_np, columns=x_ord_feats)
    x_filled_df['latitude'] = x['latitude']
    x_filled_df['longitude'] = x['longitude']

    return x_filled_df

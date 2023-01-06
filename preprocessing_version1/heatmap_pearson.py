import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# helper function to make a correlation heatmap of the categorical features
# accepts dataframe, list, and float for threshold (0 < threshold < 1)
def heatmap(x, y, threshold):
    df_x = x
    features = list(x.columns) + ['target']
    # (the target will be appended as the last row/col of the heatmap)
    x = x.to_numpy()

    # convert y with encoding (0,1,2 for the three strings)
    # because numeric values are needed for pearsonr later
    def encoding(element):
        if element == 'functional':
            return 0
        elif element == 'non functional':
            return 1
        elif element == 'functional needs repair':
            return 2
        else:
            return None

    y = list(map(encoding, y))  # makes the list numeric
    y = np.c_[y]  # need a 2D column vector

    not_continuous = []
    for col in range(np.shape(x)[1]):
        if not type(x[0, col]) == (int or float):
            not_continuous.append(int(col))

    temp_features = []
    for index, elem in enumerate(features):
        if index not in not_continuous:
            temp_features.append(elem)
    features = temp_features  # update the features

    x = np.delete(x, not_continuous, axis=1)

    # concatenate the target (y) as the final column of xTrain
    x_and_target = np.concatenate((x, y), axis=1)
    num_entries, num_features = np.shape(x_and_target)[0], np.shape(x_and_target)[1]

    # initialize the matrix that will become the heatmap
    pearson_coeff_mat = np.zeros([num_features, num_features])

    # fill up the Pearson Coefficient Matrix
    for row in range(num_features):
        for col in range(num_features):

            if not (row == col):
                # iterate through combinations of features
                feat1 = list(x_and_target[:, row])
                feat2 = list(x_and_target[:, col])

                # using the built-in pearson correlation coefficient
                corr, _ = pearsonr(feat1, feat2)
                pearson_coeff_mat[row, col] = corr
            else:
                pearson_coeff_mat[row, col] = None  # eliminate diagonal

    # convert back to dataframe
    pearson_coeff_mat = pd.DataFrame(pearson_coeff_mat, columns=features, index=features)

    # helper method to plot the heatmap (called below)
    def plot_heatmap(matrix, threshold, features):
        mat_rows = np.shape(matrix)[0]
        mat_cols = np.shape(matrix)[1]

        # create the true heatmap on the continuous (numerical) data
        sns.heatmap(matrix, cmap='coolwarm', annot=True)
        plt.title('True Pearson Correlation Heatmap')
        plt.show()

        # iterate through the matrix and decide based on the threshold
        matrix = matrix.to_numpy()
        for row in range(mat_rows):
            for col in range(mat_cols):
                if abs(matrix[row, col]) >= threshold:
                    matrix[row, col] = 1
                else:
                    matrix[row, col] = 0

        # find which features didn't meet the threshold
        to_drop = []
        for feature in range(len(features) - 2):  # exclude the target
            if matrix[np.shape(matrix)[0] - 1, feature] == 0:
                # didn't meet threshold for correlation with the target
                to_drop.append(str(features[feature]))

        # convert back to df
        matrix = pd.DataFrame(matrix, columns=features, index=features)

        # create the threshold-deciding heatmap
        sns.heatmap(matrix)
        plt.title('Heatmap, Threshold = ' + str(threshold))
        plt.show()

        return to_drop

    # call plot_heatmap on the pearson_coeff_mat, return the features that failed
    to_drop = plot_heatmap(pearson_coeff_mat, threshold, features)
    # drop from the dataframe, and return the datasets
    df_x = df_x.drop(to_drop, axis=1)
    return df_x  # , pd.DataFrame(y, columns=['label'])

import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Optimization functions

def goRansac(X, y, thresh, show_plot=False, label = ("Input", "Response")):
    # Fit line using all data
    # lr = linear_model.LinearRegression()
    # lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor(residual_threshold=thresh)
    # print(X.shape)
    # print(y.shape)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    # line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    # Compare estimated coefficients
    # print("Estimated coefficients (true, linear regression, RANSAC):")
    # print(lr.coef_, ransac.estimator_.coef_)

    if show_plot:
        lw = 2
        plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                    label='Inliers')
        plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
                    label='Outliers')
        # plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear fit')
        plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
                 label='RANSAC')
        plt.legend(loc='lower right')
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.show()
        # plt.savefig('output/outliers_detection.png')
        plt.pause(0.1)

    return inlier_mask


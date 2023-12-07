import argparse

import numpy as np
from osgeo import gdal, gdal_array

import models


def preprocessing(file: str):
    # Init gdal package
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in raster image from Wei Hai city
    img_ds = gdal.Open(file, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    rows = img.shape[0]
    cols = img.shape[1]

    # Check rows, cols and shape
    print(rows)
    print(cols)
    print(img.shape)
    print(new_shape)

    # X as image data
    X = img[:, :, :7].reshape(new_shape)
    return X, img, rows, cols


def Create_manual_labels(rows, cols):
    classes = {'building': 0, 'water': 1, 'vegetation': 2}
    n_classes = len(classes)
    supervised = (n_classes) * np.ones(shape=(rows, cols))
    supervised[525:540, 670:675] = classes['building']
    supervised[266:282, 355:385] = classes['building']
    supervised[518:522, 490:510] = classes['vegetation']
    supervised[472:480, 310:340] = classes['vegetation']
    supervised[200:220, 570:650] = classes['vegetation']
    supervised[280:294, 770:810] = classes['vegetation']
    supervised[0:100, 0:100] = classes['water']
    y = supervised.ravel()
    train = np.flatnonzero(supervised < n_classes)
    test = np.flatnonzero(supervised == n_classes)
    return train, test, y


if __name__ == '__main__':
    # hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm_type', type=str, default='supervised',
                        help='algorithm type: supervised, unsupervised')
    parser.add_argument('--model', type=str, default='SVM',
                        help='model name: K_mean, SVM, SGD, DT, Gaussian_NB, Random_Forest, ANN')
    parser.add_argument('--n_clusters', type=int, default=5, help='K_mean clusters, only work in K_mean algorithm')
    parser.add_argument('--reference_show', type=bool, default=True, help='whether show reference')
    args = parser.parse_args()

    # Get file path
    file_path = './dataset/2005_0510_Subset_atm.bsq'
    X, image, rows, cols = preprocessing(file_path)

    # Create manual label
    train, test, y = Create_manual_labels(rows, cols)

    # Run algorithms
    models.Algorithm_type(args.algorithm_type, args.model, X, image, args.n_clusters, train, test, y, rows, cols,
                          args.reference_show)

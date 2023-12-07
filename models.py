import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def Algorithm_type(algorithm_type: str, model: str, X, image, n_clusters, train, test, y, rows, cols,
                   reference_show: bool):
    if algorithm_type == 'supervised':
        Supervised_Algorithm(model, X, train, test, y, rows, cols, reference_show)
    elif algorithm_type == 'unsupervised':
        Unsupervised_Algorithm(model, X, image, n_clusters)
    else:
        print('Unknown algorithm type!!!')


def Unsupervised_Algorithm(model: str, X, image, n_clusters):
    if model == 'K_mean':
        k_means = cluster.KMeans(n_clusters=n_clusters)
        k_means.fit(X)

        # X_cluster getting data
        X_cluster = k_means.labels_
        X_cluster = X_cluster.reshape(image[:, :, 0].shape)

        import matplotlib.pyplot as plt

        print(X_cluster.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(X_cluster)
        plt.title('k-mean', fontsize=12, color='r')
        plt.show()
    else:
        print("Unknown unsupervised algorithm!!!")


def Supervised_Algorithm(model: str, X, train, test, y, rows, cols, reference_show: bool):
    # X_refer preprocessing
    X_refer = np.load('X_refer.npy')
    X_refer_image = X_refer
    X_refer = X_refer.astype(float)
    result_1d_shape = (765 * 895, 1)
    X_refer = X_refer.reshape(result_1d_shape)

    # Show reference
    if reference_show:
        print(X_refer_image.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(X_refer_image)
        plt.title('reference', fontsize=12, color='r')
        plt.show()

    if model == 'SVM':
        clf = svm.SVC()
        clf.fit(X[train], y[train])
        y[test] = clf.predict(X[test])
        supervised = y.reshape(rows, cols)

        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('SVM', fontsize=12, color='r')
        plt.show()

        supervised = supervised.reshape(result_1d_shape)
        conf_mat = confusion_matrix(X_refer, supervised)
        class_report = classification_report(X_refer, supervised)

        print("Confusion Matrix of SVM")
        print(conf_mat)
        print("Classification Report of SVM")
        print(class_report)

    elif model == 'SGD':
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        clf.fit(X[train], y[train])
        y[test] = clf.predict(X[test])
        supervised = y.reshape(rows, cols)
        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('SGD', fontsize=12, color='r')
        plt.show()

    elif model == 'DT':
        clf = tree.DecisionTreeClassifier()
        clf.fit(X[train], y[train])
        y[test] = clf.predict(X[test])
        supervised = y.reshape(rows, cols)

        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('decision tree', fontsize=12, color='r')
        plt.show()

        supervised = supervised.reshape(result_1d_shape)
        conf_mat = confusion_matrix(X_refer, supervised)
        class_report = classification_report(X_refer, supervised)

        print("Confusion Matrix of decision tree")
        print(conf_mat)
        print("Classification Report of decision tree")
        print(class_report)

    elif model == 'Gaussian_NB':
        gnb = GaussianNB()
        y[test] = gnb.fit(X[train], y[train]).predict(X[test])
        supervised = y.reshape(rows, cols)

        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('GaussianNB', fontsize=12, color='r')
        plt.show()

        supervised = supervised.reshape(result_1d_shape)
        conf_mat = confusion_matrix(X_refer, supervised)
        class_report = classification_report(X_refer, supervised)

        print("Confusion Matrix of GaussianNB")
        print(conf_mat)
        print("Classification Report of GaussianNB")
        print(class_report)

    elif model == "Random_Forest":

        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X[train], y[train])
        y[test] = clf.predict(X[test])
        supervised = y.reshape(rows, cols)

        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('randomforest', fontsize=12, color='r')
        plt.show()

        supervised = supervised.reshape(result_1d_shape)
        conf_mat = confusion_matrix(X_refer, supervised)
        class_report = classification_report(X_refer, supervised)

        print("Confusion Matrix of random forest")
        print(conf_mat)
        print("Classification Report of random forest")
        print(class_report)

    elif model == 'ANN':
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        clf.fit(X[train], y[train])
        y[test] = clf.predict(X[test])
        supervised = y.reshape(rows, cols)

        print(supervised.shape)
        plt.figure(figsize=(7, 7))
        plt.imshow(supervised)
        plt.title('neural network', fontsize=12, color='r')
        plt.show()

        supervised = supervised.reshape(result_1d_shape)
        conf_mat = confusion_matrix(X_refer, supervised)
        class_report = classification_report(X_refer, supervised)

        print("Confusion Matrix of neural network")
        print(conf_mat)
        print("Classification Report of neural network")
        print(class_report)

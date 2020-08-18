# Author: Amin Mansouri
# Time: 22:38, 10 December 2018

# Library inclusion

from itertools import cycle
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
import pickle

# To use os.path module, it should be imported. (But os is not a package itself.)
import os.path


# image = Image.open('21.png')
# image.show()

# ################################################# Functions #################################################


def load_images_from_folder(folder):
    images = []

    # os.listdir('path') returns name of the folders and files in 'path'.

    # for loop is on the name of the files in folder. Hence folder should be something like ./training/4 so that
    # os.listdir returns name of all png files inside this folder
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        images.append(img)

    # Output images are in a list
    return images


# Also glob can be used to load *.extension i.e. *.png


def multi_class_confusion_matrix_conversion(cm):
    true_positive = []
    for index in range(0, len(cm)):
        true_positive.append(cm[index][index])
    false_negative = np.sum(cm, axis=1, dtype=int) - true_positive
    false_positive = np.sum(cm, axis=0, dtype=int) - true_positive
    true_negative = np.sum(cm) - (true_positive + false_negative + false_positive)
    return true_positive, true_negative, false_positive, false_negative


def roc_curve_plot(y_test, y_pred, n_classes, title):
    y_score = y_pred
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_score))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(pd.get_dummies(y_test)).ravel(),
                                              np.array(pd.get_dummies(y_score)).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    return


# ################################################# Loading Data #################################################

# 0, 1, 2 respectively correspond to selecting PCA, LDA and GDA. To get the results for each of those, please simply
# change this selector and run the code again.
selector = 0
n_classes = 10

data = []

for i in tqdm(range(10)):
    data_temp = load_images_from_folder('./training/' + str(i))
    data.append(data_temp)

array_data = []
for i in tqdm(range(len(data))):
    class_data = data[i]
    for j in range(len(data[i])):
        temp = np.asarray(class_data[j])
        array_data.append(temp)

array_data = np.asarray(array_data)

data_labels = []
for i in range(10):
    temp = list(np.zeros(len(data[i])) + i)
    data_labels = data_labels + temp

# If you ever read a file using csv_read you can quickly identify anomalies using data.describe()! Such simple that is.

# data_labels = np.asarray(data_labels)
# To use and visualize images as color coded 2-D arrays, use the following piece of code for each element of the list.
# A = np.asarray(a[1])

# Ro visualize images as image (not an array) use the following code
plt.imshow(array_data[1])
plt.show()

# ################################ Training and Validation Partitioning (%20 and %80) ################################

train_img, test_img, train_lbl, test_lbl = train_test_split(array_data, data_labels, test_size=1 / 5.0,
                                                            random_state=0)

# ################################################# Denoising #################################################

# Median Filtering

train_img = ndimage.median_filter(train_img, 5)
test_img = ndimage.median_filter(test_img, 5)

# plt.imshow(im_med)
# plt.show()

# ################################################ Dimension Reduction ################################################

# Reshaping

train_img = train_img.reshape([len(train_img), 784])
test_img = test_img.reshape([len(test_img), 784])

# Standardizing the data

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)

pickle.dump(scaler, open('Scaler', 'wb'))
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# ####################### PCA #######################

# Making an instance of the Model
pca = PCA(.5)

pca.fit(train_img)

train_img_pca = pca.transform(train_img)
test_img_pca = pca.transform(test_img)
print('number of principal components preserved after data reduction is:', pca.n_components_)

# ####################### LDA #######################

clf = LinearDiscriminantAnalysis()
clf.fit(train_img, train_lbl)

train_img_lda = clf.fit_transform(train_img, train_lbl)
test_img_lda = clf.fit_transform(test_img, test_lbl)

# ####################### GDA #######################

clf_gda = QuadraticDiscriminantAnalysis()
clf_gda.fit(train_img, train_lbl)

train_img_gda = ...
test_img_gda = ...

# ################################################# Classifiers #################################################

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',
                'class 8', 'class 9']

# ################################################# SVM Classifier #################################################
c = 5
gamma = 0.05

decision_function = 'ovo'

if decision_function == 'ovr':
    one_versus_all = True
elif decision_function == 'ovo':
    one_versus_all = False

svm_model = svm.SVC(verbose=False, gamma=gamma, C=c, decision_function_shape=decision_function)

if selector == 0:  # PCA
    if one_versus_all:
        # Model Fitting
        svm_model.fit(train_img_pca, train_lbl)
        # Calculating the Score
        score = svm_model.score(test_img_pca, test_lbl)
        # Predicting Labels
        predicted_label = svm_model.predict(test_img_pca)
        # Calculating Confusion Matrix
        SVM_CM_ovr_PCA = confusion_matrix(test_lbl, predicted_label)
        print('score for PCA data reduction and SVM classifier (One versus all) is:', score)
        # Getting the summary for precision, recall, f1 score
        report = classification_report(test_lbl, predicted_label, target_names=target_names)
        # Getting the summary for accuracy and specificity
        TP, TN, FP, FN = multi_class_confusion_matrix_conversion(SVM_CM_ovr_PCA)
        TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        specificity = TN/(TN + FP)
        # Plotting ROC curves
        roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for SVM-one versus all-PCA')
        # Saving the model for future use
        pickle.dump(svm_model, open('SVM-ov-all-PCA', 'wb'))

    else:
        # Model Fitting
        svm_model.fit(train_img_pca, train_lbl)
        # Calculating the Score
        score = svm_model.score(test_img_pca, test_lbl)
        # Predicting Labels
        predicted_label = svm_model.predict(test_img_pca)
        # Calculating Confusion Matrix
        SVM_CM_ovo_PCA = confusion_matrix(test_lbl, predicted_label)
        print('score for PCA data reduction and SVM classifier (One versus one) is:', score)
        # Getting the summary for precision, recall, f1 score
        report = classification_report(test_lbl, predicted_label, target_names=target_names)
        # Getting the summary for accuracy and specificity
        TP, TN, FP, FN = multi_class_confusion_matrix_conversion(SVM_CM_ovo_PCA)
        TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        # Plotting ROC curves
        roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for SVM-one versus one-PCA')
        # Saving the model for future use
        pickle.dump(svm_model, open('SVM-ov-one-PCA', 'wb'))
elif selector == 1:  # LDA
    if one_versus_all:
        # Model Fitting
        svm_model.fit(train_img_lda, train_lbl)
        # Calculating the Score
        score = svm_model.score(test_img_lda, test_lbl)
        # Predicting Labels
        predicted_label = svm_model.predict(test_img_lda)
        # Calculating Confusion Matrix
        SVM_CM_ovr_LDA = confusion_matrix(test_lbl, predicted_label)
        print('score for LDA data reduction and SVM classifier (One versus all) is:', score)
        # Getting the summary for precision, recall, f1 score
        report = classification_report(test_lbl, predicted_label, target_names=target_names)
        # Getting the summary for accuracy and specificity
        TP, TN, FP, FN = multi_class_confusion_matrix_conversion(SVM_CM_ovr_LDA)
        TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        # Plotting ROC curves
        roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for SVM-one versus all-LDA')
        # Saving the model for future use
        pickle.dump(svm_model, open('SVM-ov-all-LDA', 'wb'))
    else:
        # Model Fitting
        svm_model.fit(train_img_lda, train_lbl)
        # Calculating the Score
        score = svm_model.score(test_img_lda, test_lbl)
        # Predicting Labels
        predicted_label = svm_model.predict(test_img_lda)
        # Calculating Confusion Matrix
        SVM_CM_ovo_LDA = confusion_matrix(test_lbl, predicted_label)
        print('score for LDA data reduction and SVM classifier (One versus one) is:', score)
        # Getting the summary for precision, recall, f1 score
        report = classification_report(test_lbl, predicted_label, target_names=target_names)
        # Getting the summary for accuracy and specificity
        TP, TN, FP, FN = multi_class_confusion_matrix_conversion(SVM_CM_ovo_LDA)
        TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        # Plotting ROC curves
        roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for SVM-one versus all-LDA')
        # Saving the model for future use
        pickle.dump(svm_model, open('SVM-ov-one-LDA', 'wb'))
# elif selector == 2:  # GDA
#     svm_model.fit(train_img_gda, train_lbl)
#     score = svm_model.score(test_img_gda, test_lbl)
#     print(score)
# ################################################# K-NN Classifier #################################################

neigh = KNeighborsClassifier(n_neighbors=5)

if selector == 0:  # PCA
    # Model Fitting
    neigh.fit(train_img_pca, train_lbl)
    # Predicting Labels
    predicted_label = neigh.predict(test_img_pca)
    # Calculating Confusion Matrix
    KNN_CM_PCA = confusion_matrix(test_lbl, predicted_label)
    # Calculating the Score
    print('score for PCA data reduction and KNN classifier is:', neigh.score(test_img_pca, test_lbl))
    # Getting the summary for precision, recall, f1 score
    report = classification_report(test_lbl, predicted_label, target_names=target_names)
    # Getting the summary for accuracy and specificity
    TP, TN, FP, FN = multi_class_confusion_matrix_conversion(KNN_CM_PCA)
    TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    # Plotting ROC curves
    roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for KNN-PCA')
    # Saving the model for future use
    pickle.dump(neigh, open('KNN-PCA', 'wb'))
elif selector == 1:  # LDA
    # Model Fitting
    neigh.fit(train_img_lda, train_lbl)
    # Predicting Labels
    predicted_label = neigh.predict(test_img_lda)
    # Calculating Confusion Matrix
    KNN_CM_LDA = confusion_matrix(test_lbl, predicted_label)
    # Calculating the Score
    print('score for LDA data reduction and KNN classifier is:', neigh.score(test_img_lda, test_lbl))
    # Getting the summary for precision, recall, f1 score
    report = classification_report(test_lbl, predicted_label, target_names=target_names)
    # Getting the summary for accuracy and specificity
    TP, TN, FP, FN = multi_class_confusion_matrix_conversion(KNN_CM_LDA)
    TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    # Plotting ROC curves
    roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for KNN-LDA')
    # Saving the model for future use
    pickle.dump(neigh, open('KNN-LDA', 'wb'))
# elif selector == 2:  # GDA
#     neigh.fit(train_img_gda, train_lbl)
#     print(neigh.score(test_img_gda, test_lbl))

# ############################################# Random Forest Classifier #############################################

rf = RandomForestRegressor(n_estimators=50)

if selector == 0:  # PCA
    # Model Fitting
    rf.fit(train_img_pca, train_lbl)
    # Predicting Labels
    predicted_label = np.round(rf.predict(test_img_pca))
    # Calculating Confusion Matrix
    RF_CM_PCA = confusion_matrix(test_lbl, predicted_label)
    # Calculating the Score
    print('score for PCA data reduction and Random Forest classifier is:', rf.score(test_img_pca, test_lbl))
    # Getting the summary for precision, recall, f1 score
    report = classification_report(test_lbl, predicted_label, target_names=target_names)
    # Getting the summary for accuracy and specificity
    TP, TN, FP, FN = multi_class_confusion_matrix_conversion(RF_CM_PCA)
    TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    # Plotting ROC curves
    roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for RandomForest-PCA')
    # Saving the model for future use
    pickle.dump(rf, open('RandomForest-PCA', 'wb'))
elif selector == 1:  # LDA
    # Model Fitting
    rf.fit(train_img_lda, train_lbl)
    # Predicting Labels
    predicted_label = rf.predict(test_img_lda)
    # Calculating Confusion Matrix
    RF_CM_LDA = confusion_matrix(test_lbl, predicted_label)
    # Calculating the Score
    print('score for PCA data reduction and Random Forest classifier is:', rf.score(test_img_lda, test_lbl))
    # Getting the summary for precision, recall, f1 score
    report = classification_report(test_lbl, predicted_label, target_names=target_names)
    # Getting the summary for accuracy and specificity
    TP, TN, FP, FN = multi_class_confusion_matrix_conversion(RF_CM_LDA)
    TPTNFPFN = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    # Plotting ROC curves
    roc_curve_plot(test_lbl, predicted_label, n_classes, 'ROC curve for RandomForest-LDA')
    # Saving the model for future use
    pickle.dump(rf, open('RandomForest-LDA', 'wb'))
# elif selector == 2:  # GDA
#     rf.fit(train_img_gda, train_lbl)
#     print(rf.score(test_img_gda, test_lbl))
#
print('All Done!')
# ############################################# The End! #############################################

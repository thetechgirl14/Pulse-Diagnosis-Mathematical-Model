import numpy as np
import datagen as dg
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib
from matplotlib import pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
    print("Done Importing! \n")
    
    data_gen = dg.gen_data('set1')
    train_feat, train_class = data_gen.get_data('train')
    test_feat, test_class = data_gen.get_data('test')
    val_feat, val_class = data_gen.get_data('val')
    
    print("Done getting data! \n")
    
    print("Creating model object.... \n")
    clf = svm.SVC(C = 10.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    
    print("Getting Params... \n")
    params = clf.get_params(deep=True)
    #print params
    Cs = [1.0,10.0,100.0]
    gammas = [0.001,0.01,0.1,1.0]
    kernels = ['poly', 'rbf']
    params_g = {'C': Cs, 'gamma': gammas}
    
    clf.fit(train_feat,train_class)
    
    print("Making Predictions.... \n")
    predictions_val = clf.predict(val_feat)
    predictions_test = clf.predict(test_feat)
    predictions_train = clf.predict(train_feat)
    print("Validation accuracy is ", accuracy_score(val_class,predictions_val)*100, "\n")
    print("Test accuracy is ", accuracy_score(test_class,predictions_test)*100, "\n")
    print("Train accuracy is ", accuracy_score(train_class,predictions_train)*100, "\n")
    filename_svm = 'svm_trained.sav'
    joblib.dump(clf, filename_svm)
    title = "Learning Curves SVM"
    estimator = clf

"""
Best Parameter Combination:

SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
  """
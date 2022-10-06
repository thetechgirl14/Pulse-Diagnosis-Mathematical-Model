import numpy as np
import datagen as dg
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
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

    clf = MLPClassifier(hidden_layer_sizes=(100,10,10,50), max_iter=600, alpha=0.1, learning_rate='constant',
                    solver='lbfgs', verbose=0, tol=1e-4, random_state=1, learning_rate_init=.05)

    clf.fit(train_feat, train_class)
    print("Making Predictions.... \n")
    predictions_test = clf.predict(test_feat)
    predictions_val = clf.predict(val_feat)
    predictions_train = clf.predict(train_feat)
    print("Validation accuracy is ", accuracy_score(val_class,predictions_val)*100, "\n")
    print("Test Accuracy is ", accuracy_score(test_class,predictions_test)*100, "\n")
    print("Train accuracy is ", accuracy_score(train_class,predictions_train)*100, "\n")
    filename_ann = 'ann_trained.sav'
    joblib.dump(clf, filename_ann)

    print("Best Parameter Combination: \n")
    params = clf.get_params(deep=True)
    print(params)

    title = "Learning Curves (MLP Classifier Neural Networks)(Set 1)"
    estimator = clf
    plot_learning_curve(estimator, title, train_feat, train_class, ylim=(0.4, 1.01), cv=None, n_jobs=4)
    plt.show() 

"""
Best Parameter Combination:

{'beta_1': 0.9, 'warm_start': False, 'beta_2': 0.999, 'shuffle': True, 'verbose': 0, 'nesterovs_momentum': True, 'hidden_layer_sizes': (100, 10, 10, 50), 'epsilon': 1e-08, 'activation': 'relu', 'max_iter': 600, 'batch_size': 'auto', 'power_t': 0.5, 'random_state': 1, 'learning_rate_init': 0.05, 'tol': 0.0001, 'validation_fraction': 0.1, 'alpha': 0.1, 'solver': 'lbfgs', 'momentum': 0.9, 'learning_rate': 'constant', 'early_stopping': False}
"""
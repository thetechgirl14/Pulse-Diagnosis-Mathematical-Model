import numpy as np
import datagen as dg
from sklearn.tree import DecisionTreeClassifier
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

def dt(dataset, param_printer=False):
    if dataset == 'set1':
        print("Using Dataset1 \n")
        
        data_gen = dg.gen_data('set1')
        train_feat, train_class = data_gen.get_data('train')
        test_feat, test_class = data_gen.get_data('test')
        val_feat, val_class = data_gen.get_data('val')
        
        print("Done getting data! \n")
        
        print("Fitting Models.... \n")
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
        clf_gini.fit(train_feat, train_class)
        
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
        clf_entropy.fit(train_feat, train_class)
        
        print("Making Predictions.... \n")
        predictions_gini = clf_gini.predict(test_feat)
        predictions_ent = clf_entropy.predict(test_feat)
        predictions_val_gini = clf_gini.predict(val_feat)
        predictions_train_gini = clf_gini.predict(train_feat)
        predictions_val_ent = clf_entropy.predict(val_feat)
        predictions_train_ent = clf_entropy.predict(train_feat)
        filename_gini = 'dt_gini_trained.sav'
        joblib.dump(clf_gini, filename_gini)
        filename_entropy = 'dt_entropy_trained.sav'
        joblib.dump(clf_entropy, filename_entropy)
        
        print("Test Accuracy for Gini is ", accuracy_score(test_class,predictions_gini)*100, "\n")
        print("Test Accuracy for Info Gain is ", accuracy_score(test_class,predictions_ent)*100, "\n")
        print("Validation accuracy for Gini is ", accuracy_score(val_class,predictions_val_gini)*100, "\n")
        print("Validation accuracy for Info Gain is ", accuracy_score(val_class,predictions_val_ent)*100, "\n")
        print("Train accuracy for Gini is ", accuracy_score(train_class,predictions_train_gini)*100, "\n")
        print("Train accuracy for Info Gain is ", accuracy_score(train_class,predictions_train_ent)*100, "\n")
        
        if param_printer == True:
            print("Best Parameter Combination Gini: \n")
            params_gini = clf_gini.get_params(deep=True)
            print(params_gini)
        
            print("Best Parameter Combination Info Gain: \n")
            params_entropy = clf_entropy.get_params(deep=True)
            print(params_entropy)
        
        title = "Learning Curves (Decision Trees Gini)(Set 1)"
        estimator = clf_gini
        plot_learning_curve(estimator, title, train_feat, train_class, ylim=(0.4, 1.01), cv=None, n_jobs=4)
        
        title = "Learning Curves (Decision Trees Info Gain)(Set 1)"
        estimator = clf_entropy
        plot_learning_curve(estimator, title, train_feat, train_class, ylim=(0.4, 1.01), cv=None, n_jobs=4)
        plt.show()


if __name__ == '__main__':
    dt(dataset='set1')
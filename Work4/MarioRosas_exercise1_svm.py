#############################################################
#############################################################
#############################################################


import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay

if __name__ == "__main__":
    import pylab as pl

    def plot_training_data_with_decision_boundary(clf, X, y):
        # Train the SVC
        # clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

        # Settings for plotting
        _, ax = plt.subplots(figsize=(4, 3))
        x_min, x_max, y_min, y_max = -3, 3, -3, 3
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Plot decision boundary and margins
        common_params = {"estimator": clf, "X": X, "ax": ax}
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="predict",
            plot_method="pcolormesh",
            alpha=0.3,
        )
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )

        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=250,
            facecolors="none",
            edgecolors="k",
        )
        # Plot samples by color and add legend
        ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

        _ = plt.show()

    def plot_support_vectors(X_train, y_train, svc_model):
        plt.figure(figsize=(10, 8))
        # Plotting our two-features-space
        sns.scatterplot(x=X_train[:, 0], 
                        y=X_train[:, 1], 
                        hue=y_train, 
                        s=15);
        # Constructing a hyperplane using a formula.
        w = svc_model.coef_[0]           # w consists of 2 elements
        b = svc_model.intercept_[0]      # b consists of 1 element
        x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
        y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
        # Plotting a red hyperplane
        plt.plot(x_points, y_points, c='r');
        # Encircle support vectors
        plt.scatter(svc_model.support_vectors_[:, 0],
                    svc_model.support_vectors_[:, 1], 
                    s=50, 
                    facecolors='none', 
                    edgecolors='k', 
                    alpha=.5);
        # Step 2 (unit-vector):
        w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))
        # Step 3 (margin):
        margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))
        # Step 4 (calculate points of the margin lines):
        decision_boundary_points = np.array(list(zip(x_points, y_points)))
        points_of_line_above = decision_boundary_points + w_hat * margin
        points_of_line_below = decision_boundary_points - w_hat * margin
        # Plot margin lines
        # Blue margin line above
        plt.plot(points_of_line_above[:, 0], 
                 points_of_line_above[:, 1], 
                 'b--', 
                 linewidth=2)
        # Green margin line below
        plt.plot(points_of_line_below[:, 0], 
                 points_of_line_below[:, 1], 
                 'g--',
                 linewidth=2)
        plt.show()


    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def run_svm_dataset1():
        print(f"Dataset 1")
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

         #### 
        # Write here your SVM code and choose a linear kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        correct_preds = [1 for i,p in enumerate(predictions) if p == y_test[i]]
        print(f"Total predictions: {len(predictions)}, Correct predictions {len(correct_preds)}")
        plot_support_vectors(X_train, y_train, clf)

    def run_svm_dataset2():
        print(f"Dataset 2")
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and choose a linear kernel with the best C pparameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
        svc = SVC()
        clf = GridSearchCV(svc, {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[1,5,10,15,20,50,100]})
        clf.fit(X_train, y_train)
        bclf = clf.best_estimator_
        predictions = clf.predict(X_test)
        correct_preds = [1 for i,p in enumerate(predictions) if p == y_test[i]]
        print(f"Best params are: {clf.best_params_}")
        print(f"Total predictions: {len(predictions)}, Correct predictions {len(correct_preds)}")
        # plot_support_vectors(X_train, y_train, bclf)
        # plot_training_data_with_decision_boundary(bclf, X_train, y_train)



    def run_svm_dataset3():
        print(f"Dataset 3")
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and use a gaussian kernel 
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####

        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        correct_preds = [1 for i,p in enumerate(predictions) if p == y_test[i]]
        print(f"Total predictions: {len(predictions)}, Correct predictions {len(correct_preds)}")
        # plot_support_vectors(X_train, y_train, clf)


#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS      
    run_svm_dataset1()   # data distribution 1
    run_svm_dataset2()   # data distribution 2
    run_svm_dataset3()   # data distribution 3

#############################################################
#############################################################
#############################################################




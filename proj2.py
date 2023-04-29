import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA


# Calculate the measures of performance of a model
def calc_measures(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("accuracy_score:", accuracy)
    print("precision_score:", macro_recall)
    print("recall_score:", macro_precision)
    print("f1_score:", macro_f1)

    return accuracy, macro_recall, macro_precision, macro_f1

# Select features based on [3]
# sort features by some score and select the best, doesnt account for inter feature selection
@ignore_warnings(category=ConvergenceWarning)
def select_features(X_train, y_train, k=25):
    # Initialize logistic regression model
    model = LogisticRegression(penalty=None, solver='lbfgs', multi_class='multinomial', max_iter=1000)

    # Loop over each feature and compute its score
    scores = []
    for j in range(X_train.shape[1]):
        if (j == 0) or ((j + 1) % 10 == 0):
            print("iteration: " + str(j+1) + "/" + str(X_train.shape[1]))
        # Select all features except for j
        X_sel = X_train.drop(X_train.columns[j], axis=1)
        # Fit the model and compute conditional probabilities
        model.fit(X_sel, y_train)
        probas = softmax(model.predict_log_proba(X_sel), axis=1)
        # Compute the conditional likelihood
        Lj = np.sum(np.log(probas[np.arange(len(y_train)), y_train]))
        # Compute the score as the negative of the conditional likelihood
        scores.append(-Lj)

    # Select the top-k features with the highest scores
    selected_features = np.argsort(scores)[:k]
    return selected_features


# KNN Classification Model - euclidean distance
def KNNClassification(x, y, test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y)
    y_predict = knn.predict(test)
    return y_predict


# Naive Bayes Classification Model
def NBClassification(x, y, test):
    model = GaussianNB()
    model.fit(x, y)
    y_predict = model.predict(test)
    return y_predict

# Select features with PCA, then evaluate with KNN and NB
def ApplyPCA(X_train, X_test, y_test, k=5):
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train) # used on the training data so that we can scale the training data and also learn the scaling parameters of that data
    X_test_pca = pca.transform(X_test) # descobrir como ta a fazer o fit da data da pra escolher os parametros
    v = pca.explained_variance_ratio_
    print("explained variance ratio:")
    print(v)
    print(np.sum(v))

    y_pred = KNNClassification(X_train_pca, y_train, X_test_pca, 5)
    print("\nKNN " + str(k) + " features")
    acc, r, p, f1 = calc_measures(y_test, y_pred)

    y_pred = NBClassification(X_train_pca, y_train, X_test_pca)
    print("\nNB " + str(k) + " features")
    accNB, rNB, pNB, f1NB = calc_measures(y_test, y_pred)

    return acc, r, p, f1, accNB, rNB, pNB, f1NB, v






# pode nao haver ortogonalidade entre os componentes explained variance ratio não se adequa
# podemos computar a variancia total https://github.com/scikit-learn/scikit-learn/issues/11512
def get_explain_var_SPCA(x_train, x_test, y_test, s_level, n_components):

    spca = SparsePCA(n_components=n_components, alpha=s_level)
    print(n_components)
    #print("Before fit x_train")
    spca.fit(x_train)
    #print("After fit x_train")
    Xc = x_train - x_train.mean(axis=0)  # center data

    P = spca.components_.T  # loadings

    #print("Before transform x_train")
    T = Xc @ P @ np.linalg.pinv(P.T @ P)
    #print("x_train tranformation computed")


    explained_variance = np.trace(P @ T.T @ T @ P.T)
    total_variance = np.trace(Xc.T @ Xc)


#    x_stest = spca.transform(x_test)
#    t_spca = spca.transform(x_train)





    return explained_variance / total_variance




def get_data():
    # Get the data
    data = pd.read_csv("data/AcousticFeatures.csv")
    train_data = data.sample(frac=0.75, random_state=200)
    test_data = data.drop(train_data.index)
    X_train = train_data.iloc[:, 1:]
    X_test = test_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    y_test = test_data.iloc[:, 0]

    return X_train, X_test, y_train, y_test

def main():

    x_train, x_test, y_train, y_test = get_data()

    sparse_levels = [i/100 for i in range(1, 11)]
    sparse_levels = [0.2, 0.3]
    max_n_components = 7
    print_spca_graphs(x_train, x_test, y_test, sparse_levels, max_n_components)







def print_spca_graphs(x_train, x_test, y_test, sparse_levels, max_n_components):
    # Initialize the list to store the explained variance ratio
    explained_var_ratios = []

    # Loop over the sparsity levels and number of components to apply Sparse PCA and compute explained variance ratio
    for s_level in sparse_levels:
        explained_var_ratios = []
        for n_comp in range(1, max_n_components):
            explained_var_ratio = get_explain_var_SPCA(x_train, x_test, y_test, s_level, n_comp)
            explained_var_ratios.append(explained_var_ratio)
            print(explained_var_ratio)

        print(explained_var_ratios)

        # Plot the explained variance ratio for each combination of sparsity level and number of components
        plt.plot(list(range(1, max_n_components)), explained_var_ratios, "-o",label=f"s_level={s_level}")
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance ratio")
    plt.title("Sparse PCA performance")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()



























# Convert the target variable to a numeric form using label encoding
le = LabelEncoder()
classes = np.unique(le.fit_transform(data.iloc[:, 0]))
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)



#3. and 4.  Select data with best 15/25 features
selected_features15 = select_features(X_train, y_train, k=15)
selected_features25 = select_features(X_train, y_train, k=25)
X_train_selected15 = X_train.iloc[:, selected_features15]
X_train_selected25 = X_train.iloc[:, selected_features25]
print(f'Selected features: {selected_features15}')
print(f'Selected features: {selected_features25}')


#5.
# Train a k-NN model using the selected features
y_pred = KNNClassification(X_train_selected15, y_train, X_test.iloc[:, selected_features15], 5)
y_pred2 = KNNClassification(X_train_selected25, y_train, X_test.iloc[:, selected_features25], 5)
print("\nKNN 15 features")
calc_measures(y_test, y_pred)
print("\nKNN 25 features")
calc_measures(y_test, y_pred2)


# Gaussian Naive Bayes model
y_pred = NBClassification(X_train_selected15, y_train, X_test.iloc[:, selected_features15])
y_pred2 = NBClassification(X_train_selected25, y_train, X_test.iloc[:, selected_features25])
print("\nNB 15 features")
calc_measures(y_test, y_pred)
print("\nNB 25 features")
calc_measures(y_test, y_pred2)


#6. PCA
print("\n\n\nPCA")
ApplyPCA(X_train, X_test, y_test, 15)
ApplyPCA(X_train, X_test, y_test, 25)


#7. Other dimensionality reduction method
# TODO



#8. Plots
n_components = range(1, X_train.shape[1]+1)
explained_variance = []
accuraciesKNN = []
accuraciesNB = []

for n in n_components:
    acc, r, p, f1, accNB, rNB, pNB, f1NB, v = ApplyPCA(X_train, X_test, y_test, n)
    accuraciesKNN.append(acc)
    accuraciesNB.append(accNB)
    if n == (X_train.shape[1]):
        explained_variance = v

plt.bar(n_components, explained_variance)
plt.title("Explained Variance")
plt.xlabel("Components")
plt.ylabel("Explained Variance")
plt.show()

plt.plot(n_components, accuraciesKNN)
plt.title("Accuracy KNN")
plt.xlabel("N Components")
plt.ylabel("Accuracy")
plt.show()

plt.plot(n_components, accuraciesNB)
plt.title("Accuracy NB")
plt.xlabel("N Components")
plt.ylabel("Accuracy")
plt.show()

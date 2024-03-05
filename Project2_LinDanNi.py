#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Project 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("/Users/dannilin/Desktop/PR/Project2/Proj2DataSet.xlsx", header=None, names=["feature1", "feature2", "label"])


# Count instances of each class
class_counts = df['label'].value_counts()

plt.figure(figsize=(10, 6))

colors = plt.cm.winter(np.linspace(0, 1, len(class_counts)))  # Generate colors

# Plot each class separately to have individual labels
for i, (class_label, count) in enumerate(class_counts.iteritems()):
    # Select rows where the class label matches
    class_data = df[df['label'] == class_label]
    
    plt.scatter(class_data['feature1'], class_data['feature2'], color=colors[i], label=f'Class {class_label} (n={count})')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Visualization')
plt.legend()
plt.show()


# In[2]:


df.describe()


# In[3]:


# Calculate variance of features
variance_values = df.var()
print("\nVariance Values:\n", variance_values)


# In[4]:


# Calculate variance for each column for each species
variance_df = df.groupby('label').var()

print(variance_df)


# In[5]:


# Within-Class Variance

sw_1 = (40/100)*1.978200 + (60/100)*1.037358
sw_2 = (40/100)*1.581348 + (60/100)*0.941975

print('Within-Class Variance')
print('Feature 1: ', np.round(sw_1,2)) 
print('Feature 2: ',np.round(sw_2,2)) 


# In[6]:


# Count the occurrences of each species
species_counts = df['label'].value_counts()

print(species_counts)


# In[7]:


# Calculate variance for each column for each species
mean_df = df.groupby('label').mean()

print(mean_df)


# In[8]:


# Between-Class Variance

sb_1 = (40/100)*(3.83148-2.10)**2 + (60/100)*(0.930722-2.10)**2
sb_2 = (40/100)*(1.181589-2.27)**2 + (60/100)*(3.002650-2.27)**2 

print('Between-Class Variance')
print('Feature 1: ', np.round(sb_1,2)) 
print('Feature 2: ',np.round(sb_2,2)) 


# ## Soft SVM

# In[9]:


import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def solve_dual_svm_problem(X, y, C):

    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = linear_kernel(X[i], X[j])
    
    P = matrix(np.outer(y,y) * K) #matrix of quadratic terms of alpha
    q = matrix(np.ones(n_samples) * -1) #linear terms of alpha

    #A and b matrices refers to the equality constraint: sum of alpha_i * y_i = 0
    A = matrix(y, (1,n_samples), 'd') 
    b = matrix(0.0) 

    #G and h matrices refers to the inequality constraint: 0 <= alpha_i <= C
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    
    # Solve QP problem
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    
    return alphas

def custom_sign(values):
    """
    Returns 1 for non-negative values and -1 for negative values.
    This adjusts the standard np.sign behavior to treat 0 as 1.
    """
    return np.where(values > 0, 1, -1)

# Function to count misclassified samples
def count_misclassified_samples(X, y, w, b):
    decisions = np.dot(X, w) + b
    predictions = custom_sign(decisions)
    misclassified = np.sum(predictions != y)
    return misclassified

def plot_decision_boundary(X, y, alphas, C):
    
    # Filter out non-zero alphas to find support vectors
    sv = alphas > 1e-3
    ind = np.arange(len(alphas))[sv]
    alphas_sv = alphas[sv]
    sv_x = X[sv]
    sv_y = y[sv]
    
    # Compute weight vector and bias
    w = np.dot(sv_x.T, alphas_sv * sv_y)
    b = np.mean(sv_y - np.dot(sv_x, w))
    
    # Plot data points and support vectors
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.scatter(sv_x[:, 0], sv_x[:, 1], s=100, facecolors='none', edgecolors='k', label=f'Support Vectors: {len(sv_x)}')
  
    
    # Plot decision boundary and margins
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], colors='k')

    # Identify and plot misclassified samples
    decisions = np.dot(X, w) + b
    predictions = custom_sign(decisions)
    misclassified_indices = np.where(predictions != y)[0]
    misclassified_count = count_misclassified_samples(X, y, w, b)
    plt.scatter(X[misclassified_indices, 0], X[misclassified_indices, 1], s=50, c='red', marker='x', label=f'Misclassified Samples: {misclassified_count}')

    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary with C={C}')
    plt.legend()
    plt.show()
    
    misclassified_count = count_misclassified_samples(X, y, w, b)
    print(f'Number of Misclassified Samples: {misclassified_count}')
    print(f'Number of Support Vectors: {len(sv_x)}')


# In[10]:


# Load the dataset
df = pd.read_excel("/Users/dannilin/Desktop/PR/Project2/Proj2DataSet.xlsx", engine='openpyxl')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#print(X)
#print(y)


# In[11]:


# Example usage
C_values = [0.1, 100]

for C in C_values:
    alphas = solve_dual_svm_problem(X, y, C)
    plot_decision_boundary(X, y, alphas, C)


# ## SMO Vs. SVM implementation

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd 


def run_timing():
    np.random.seed(100)

    time_my_model_c_01 = []
    time_sklearn_c_01 = []
    time_my_model_c_100 = []
    time_sklearn_c_100 = []

    for c in [0.1, 100]:
        for i in range(100, 1000, 100):
            
            # Generate synthetic data
            class1 = np.random.multivariate_normal([1, 3], [[1, 0], [0, 1]], i)
            class2 = np.random.multivariate_normal([4, 1], [[2, 0], [0, 2]], i)
            X_train, y_train = np.vstack((class1, class2)), np.hstack((np.ones(i), -np.ones(i)))

            # Time scikit-learn SVM
            start = time.time()
            clf_sklearn = make_pipeline(StandardScaler(), SVC(gamma='auto', C=c))
            clf_sklearn.fit(X_train, y_train)
            end = time.time()
            sklearn_duration = end - start
            if c == 0.1:
                time_sklearn_c_01.append(sklearn_duration)
            else:
                time_sklearn_c_100.append(sklearn_duration)

            # Time custom solve_dual_svm_problem
            start = time.time()
            alphas = solve_dual_svm_problem(X_train, y_train, C=c)  # This replaces the SVM class
            end = time.time()
            custom_duration = end - start
            if c == 0.1:
                time_my_model_c_01.append(custom_duration)
            else:
                time_my_model_c_100.append(custom_duration)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(range(100, 1000, 100), time_my_model_c_01, '-x', label='My own SVM implementation with C=0.1')
    plt.plot(range(100, 1000, 100), time_sklearn_c_01, '-x', label='SMO: scikit-learn SVM C=0.1')
    plt.plot(range(100, 1000, 100), time_my_model_c_100, '-x', label='My own SVM implementation with C=100')
    plt.plot(range(100, 1000, 100), time_sklearn_c_100, '-x', label='SMO: scikit-learn SVM C=100')
    plt.xlabel('Number of Samples')
    plt.ylabel('Execution Time (seconds)')
    plt.title('SVM Training Time Comparison')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_timing()


# In[ ]:





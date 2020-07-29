# Paediatric_Cancers_Classifiers
Context: Capstone project of Udacity's Machine Learning Engineer Nanodegree.

About:
1. The Python script takes a dataset from a pan-cancer analysis of paediatric cancers as inputs.
2. It builds a series of classifiers to predict cancer histotypes and trains them on the dataset comprising activities of mutational signatures, including a decision tree, a naive Bayes classifier, support vector machines, an ensemble method (Adaboost), and a multilayer perceptron.
3. It quantifies the intra-histotype variations in the dataset by hierarchical clustering.
4. It extracts latent features from the dataset by principal component analysis.

Files:
1. The Python script named 'Capstone.py' should be implemented in Python 3.5.
2. The dataset named 'nature25795-s4' must be in the same directory as the script when the latter is run.
3. The script can be run without changes. The only problem is that the three dendrograms produced by the final block of the script will be squeezed into one plot. One needs to plot the dendrograms separately again after running the script.

Modules:
1. NumPy is needed for array and matrix support.
2. Pandas is needed for data manipulation and analysis.
3. matplotlib is needed for visualisation.
4. scikit-learn is needed for most of the classification models, their optimisation, their metrics, and principal component analysis.
5. Keras and TensorFlow are needed for deep learning.
6. Scipy is needed for hierarchical clustering.

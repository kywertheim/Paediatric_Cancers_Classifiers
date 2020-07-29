#Import libraries to handle the dataset.
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklprepro

#Import a library for visualisation.
import matplotlib.pyplot as plt

#Import libraries for classification.
import sklearn.model_selection as sklmodsel
import sklearn.dummy as skldummy
import sklearn.utils as sklutils
import sklearn.metrics as sklmet
import sklearn.tree as skltree
import sklearn.naive_bayes as sklnb
import sklearn.svm as sklsvm
import sklearn.ensemble as sklensemble
import keras.models as krsmodels
import keras.layers as krslayers
import keras.optimizers as krsop
import keras.callbacks as krscb
import keras.utils.np_utils as krsutil
import keras.regularizers as krsreg

#Import libraries for clustering.
import scipy.cluster.hierarchy as scpyhierclus

#Import a library for dimensionality reduction.
import sklearn.decomposition as skldecomp

#Import the dataset, drop all unnecessary columns and rows, and rename the remaining columns.
df=pd.read_excel('nature25795-s4.xlsx', 'Table S1a')
df.drop(['Unnamed: 1', 'Unnamed: 2'], axis=1, inplace=True)
df=df.iloc[3:,]
dfheader=df.iloc[0]
df=df.iloc[1:,]
df.rename(columns=dfheader, inplace=True)
df.rename(columns={np.nan:'Cancer Type'}, inplace=True)
df['Signature T-1']=pd.to_numeric(df['Signature T-1'])
df['Signature T-2']=pd.to_numeric(df['Signature T-2'])
df['Signature T-3']=pd.to_numeric(df['Signature T-3'])
df['Signature T-4']=pd.to_numeric(df['Signature T-4'])
df['Signature T-5']=pd.to_numeric(df['Signature T-5'])
df['Signature T-6']=pd.to_numeric(df['Signature T-6'])
df['Signature T-7']=pd.to_numeric(df['Signature T-7'])
df['Signature T-8']=pd.to_numeric(df['Signature T-8'])
df['Signature T-9']=pd.to_numeric(df['Signature T-9'])
df['Signature T-10']=pd.to_numeric(df['Signature T-10'])
df['Signature T-11']=pd.to_numeric(df['Signature T-11'])

#Split the dataset into six subsets by label.
df_tall=df.loc[df['Cancer Type']=='TALL']
df_ball=df.loc[df['Cancer Type']=='BALL']
df_aml=df.loc[df['Cancer Type']=='AML']
df_nbl=df.loc[df['Cancer Type']=='NBL']
df_wt=df.loc[df['Cancer Type']=='WT']
df_os=df.loc[df['Cancer Type']=='OS']

#Make a copy of the dataset, drop the columns of features, and check that there are not any missing labels in the entries.
y=df['Cancer Type']
y.isnull().any()

#Make a copy of the dataset, drop the column of labels, and check that there are not any missing features in the entries.
x=df.drop(['Cancer Type'], axis=1)
x.isnull().any()

#Make a copy of each subset of the dataset and drop the column of labels.
x_tall=df_tall.drop(['Cancer Type'], axis=1)
x_ball=df_ball.drop(['Cancer Type'], axis=1)
x_aml=df_aml.drop(['Cancer Type'], axis=1)
x_nbl=df_nbl.drop(['Cancer Type'], axis=1)
x_wt=df_wt.drop(['Cancer Type'], axis=1)
x_os=df_os.drop(['Cancer Type'], axis=1)

#Exploratory analysis of the label column.
y.value_counts()
bar_chart=plt.figure(figsize=(6,4))
bar_chart.suptitle('Number of Entries for each Cancer Histotype')
y.value_counts().plot(kind='bar')

#Exploratory analysis of the feature columns.
x.mode()
x[x>0].min()
x['Signature T-1'].describe()
x['Signature T-2'].describe()
x['Signature T-3'].describe()
x['Signature T-4'].describe()
x['Signature T-5'].describe()
x['Signature T-6'].describe()
x['Signature T-7'].describe()
x['Signature T-8'].describe()
x['Signature T-9'].describe()
x['Signature T-10'].describe()
x['Signature T-11'].describe()
nbins=20
histograms, hist_axes=plt.subplots(nrows=4, ncols=3, figsize=(15,15))
T1his, T2his, T3his, T4his, T5his, T6his, T7his, T8his, T9his, T10his, T11his, dummy=hist_axes.flatten()
T1his.hist(x['Signature T-1'], nbins)
T1his.set_title('Signature T-1')
T2his.hist(x['Signature T-2'], nbins)
T2his.set_title('Signature T-2')
T3his.hist(x['Signature T-3'], nbins)
T3his.set_title('Signature T-3')
T4his.hist(x['Signature T-4'], nbins)
T4his.set_title('Signature T-4')
T5his.hist(x['Signature T-5'], nbins)
T5his.set_title('Signature T-5')
T6his.hist(x['Signature T-6'], nbins)
T6his.set_title('Signature T-6')
T7his.hist(x['Signature T-7'], nbins)
T7his.set_title('Signature T-7')
T8his.hist(x['Signature T-8'], nbins)
T8his.set_title('Signature T-8')
T9his.hist(x['Signature T-9'], nbins)
T9his.set_title('Signature T-9')
T10his.hist(x['Signature T-10'], nbins)
T10his.set_title('Signature T-10')
T11his.hist(x['Signature T-11'], nbins)
T11his.set_title('Signature T-11')
histograms.delaxes(hist_axes[3,2])
x['Signature T-9'].unique()
x['Signature T-9'].value_counts()

#Data preprocessing. Label encoding the label column of the dataset.
yLabelEncoder=sklprepro.LabelEncoder()
yLabelEncoder.fit(y)
yLabelEncoder.classes_
y_encoded=yLabelEncoder.transform(y)
y_decoded=yLabelEncoder.inverse_transform(y_encoded)
y_final=pd.DataFrame(y_encoded)
y_final.isnull().any()

#Data preprocessing. Logarithmic transformation and then normalisation of every feature column of the dataset.
x_log=x.copy()
x_log[x_log.columns.values]=x_log[x_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_log.isnull().any()
x_log.describe()
x_final=x_log.copy()
scaler=sklprepro.MinMaxScaler()
x_final[x_final.columns.values]=scaler.fit_transform(x_final[x_final.columns.values])
x_final.isnull().any()
x_final.describe()

#Data preprocessing. Logarithmic transformation and then normalisation of every feature column of every subset of the dataset.
x_tall_log=x_tall.copy()
x_tall_log[x_tall_log.columns.values]=x_tall_log[x_tall_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_tall_final=x_tall_log.copy()
x_tall_final[x_tall_final.columns.values]=scaler.fit_transform(x_tall_final[x_tall_final.columns.values])
x_ball_log=x_ball.copy()
x_ball_log[x_ball_log.columns.values]=x_ball_log[x_ball_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_ball_final=x_ball_log.copy()
x_ball_final[x_ball_final.columns.values]=scaler.fit_transform(x_ball_final[x_ball_final.columns.values])
x_aml_log=x_aml.copy()
x_aml_log[x_aml_log.columns.values]=x_aml_log[x_aml_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_aml_final=x_aml_log.copy()
x_aml_final[x_aml_final.columns.values]=scaler.fit_transform(x_aml_final[x_aml_final.columns.values])
x_nbl_log=x_nbl.copy()
x_nbl_log[x_nbl_log.columns.values]=x_nbl_log[x_nbl_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_nbl_final=x_nbl_log.copy()
x_nbl_final[x_nbl_final.columns.values]=scaler.fit_transform(x_nbl_final[x_nbl_final.columns.values])
x_wt_log=x_wt.copy()
x_wt_log[x_wt_log.columns.values]=x_wt_log[x_wt_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_wt_final=x_wt_log.copy()
x_wt_final[x_wt_final.columns.values]=scaler.fit_transform(x_wt_final[x_wt_final.columns.values])
x_os_log=x_os.copy()
x_os_log[x_os_log.columns.values]=x_os_log[x_os_log.columns.values].apply(lambda xi: np.log(xi+0.000001))
x_os_final=x_os_log.copy()
x_os_final[x_os_final.columns.values]=scaler.fit_transform(x_os_final[x_os_final.columns.values])

#Split the dataset into a training subset and a testing subset.
x_train, x_test, y_train, y_test=sklmodsel.train_test_split(x_final, y_final, stratify=y_final, test_size=0.25, random_state=42)

#Benchmark model for the first task. This classifier assigns a histotype to a cancer with a probability equalling its frequency in the dataset, regardless of its mutational catalogue.
benchmark_model=skldummy.DummyClassifier(strategy='stratified', random_state=42)
x_train_checked, y_train_checked=sklutils.check_X_y(x_train, y_train)
benchmark_model.fit(x_train_checked, y_train_checked)

#Decision tree model for the first task, including multiple rounds of grid search.
tree_model=skltree.DecisionTreeClassifier(random_state=42)
#tree_hyper={'max_depth':[1,5,10,15,20,25,50,100,200,400,800], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_split':[2,4,8,16,32,64]}
#tree_hyper={'max_depth':[19,20,21], 'min_samples_leaf':[1,2,3,4,5], 'min_samples_split':[2,4,8,16]}
tree_hyper={'max_depth':[18,19,20], 'min_samples_leaf':[1,2,3], 'min_samples_split':[2,3,5]}
tree_scorer=sklmet.make_scorer(sklmet.f1_score, average='macro')
tree_GSCV=sklmodsel.GridSearchCV(tree_model, tree_hyper, scoring=tree_scorer, cv=sklmodsel.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
tree_GSCV.fit(x_train, y_train)
best_tree_model=tree_GSCV.best_estimator_
best_tree_valF1=tree_GSCV.best_score_
best_tree_model.fit(x_train, y_train)

#Naive Bayes classifier for the first task.
nb_model=sklnb.GaussianNB()
nb_model.fit(x_train, y_train)

#Linear support vector machine for the first task, including multiple rounds of grid search.
svm_linear_model=sklsvm.SVC(kernel='linear', random_state=42)
#svm_linear_hyper={'C':[0.01,0.1,1,10,100]}
#svm_linear_hyper={'C':[50,75,100,125,150]}
#svm_linear_hyper={'C':[40,45,50,55,60]}
svm_linear_hyper={'C':[40,41,42,43,44,45,46,47,48,49,50]}
svm_linear_scorer=sklmet.make_scorer(sklmet.f1_score, average='macro')
svm_linear_GSCV=sklmodsel.GridSearchCV(svm_linear_model, svm_linear_hyper, scoring=svm_linear_scorer, cv=sklmodsel.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
svm_linear_GSCV.fit(x_train, y_train)
best_svm_linear_model=svm_linear_GSCV.best_estimator_
best_svm_linear_valF1=svm_linear_GSCV.best_score_
best_svm_linear_model.fit(x_train, y_train)

#Support vector machine with the polynomial kernel for the first task, including multiple rounds of grid search.
svm_poly_model=sklsvm.SVC(kernel='poly', random_state=42)
#svm_poly_hyper={'C':[0.01,0.1,1,10,100], 'degree':[1,2,3,4,5,6,7,8,9,10]}
#svm_poly_hyper={'C':[90,100,200], 'degree':[1,2,3,4,5]}
#svm_poly_hyper={'C':[200, 500, 1000, 2000], 'degree':[1,2,3,4]}
#svm_poly_hyper={'C':[1900, 2000, 3000, 4000, 8000, 20000], 'degree':[1,2,3]}
#svm_poly_hyper={'C':[6000, 8000, 10000], 'degree':[1,2,3]}
#svm_poly_hyper={'C':[9000, 10000, 12000, 13000, 14000, 15000, 16000], 'degree':[1,2,3]}
#svm_poly_hyper={'C':[14500,15000,15500], 'degree':[1,2,3]}
#svm_poly_hyper={'C':[14900,15000,15100], 'degree':[1,2,3]}
svm_poly_hyper={'C':[14800,14900,15000], 'degree':[1,2,3]}
svm_poly_scorer=sklmet.make_scorer(sklmet.f1_score, average='macro')
svm_poly_GSCV=sklmodsel.GridSearchCV(svm_poly_model, svm_poly_hyper, scoring=svm_poly_scorer, cv=sklmodsel.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
svm_poly_GSCV.fit(x_train, y_train)
best_svm_poly_model=svm_poly_GSCV.best_estimator_
best_svm_poly_valF1=svm_poly_GSCV.best_score_
best_svm_poly_model.fit(x_train, y_train)

#Support vector machine with the radial basis function kernel for the first task, including multiple rounds of grid search.
svm_rbf_model=sklsvm.SVC(kernel='rbf', random_state=42)
#svm_rbf_hyper={'C':[0.01,0.1,1,10,100], 'gamma':[0.01,0.1,1,10,100]}
#svm_rbf_hyper={'C':[100,500,1000,5000,10000,20000], 'gamma':[0.5,1,2,3,4,5]}
#svm_rbf_hyper={'C':[90,100,110], 'gamma':[0.1,0.5,1]}
#svm_rbf_hyper={'C':[98,100,105], 'gamma':[0.3,0.5,0.7]}
#svm_rbf_hyper={'C':[100,101,102,103,104,105,106,107,108,109,110], 'gamma':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
svm_rbf_hyper={'C':[104,105,106], 'gamma':[0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75]}
svm_rbf_scorer=sklmet.make_scorer(sklmet.f1_score, average='macro')
svm_rbf_GSCV=sklmodsel.GridSearchCV(svm_rbf_model, svm_rbf_hyper, scoring=svm_rbf_scorer, cv=sklmodsel.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
svm_rbf_GSCV.fit(x_train, y_train)
best_svm_rbf_model=svm_rbf_GSCV.best_estimator_
best_svm_rbf_valF1=svm_poly_GSCV.best_score_
best_svm_rbf_model.fit(x_train, y_train)

#Multilayer perceptron for the first task.
mlp_model=krsmodels.Sequential()
mlp_model.add(krslayers.Dense(8, input_dim=11, kernel_regularizer=krsreg.l2(0.00001)))
mlp_model.add(krslayers.Activation('relu'))
mlp_model.add(krslayers.Dropout(0.1))
mlp_model.add(krslayers.Dense(6, kernel_regularizer=krsreg.l2(0.00001)))
mlp_model.add(krslayers.Activation('softmax'))
sgdop=krsop.SGD(lr=0.01, decay=2e-5, momentum=0.9, nesterov=True)
mlp_model.compile(loss='categorical_crossentropy', optimizer=sgdop, metrics=['accuracy'])
early_stop=krscb.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5000, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
callbacks_list=[early_stop]
x_mlptrain, x_mlpval, y_mlptrain, y_mlpval=sklmodsel.train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
mlp_training_log=mlp_model.fit(np.array(x_mlptrain), krsutil.to_categorical(y_mlptrain), batch_size=548, epochs=1000000, shuffle=True, validation_data=(np.array(x_mlpval), krsutil.to_categorical(y_mlpval)), callbacks=callbacks_list, verbose=2)

#AdaBoost algorithm with a decision tree as the base classifier for the first task, including several rounds of grid search.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(random_state=42), random_state=42) #F1 0.728 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=2, random_state=42), random_state=42) #Validation F1 0.484 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=3, random_state=42), random_state=42) #Validation F1 0.745 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=4, random_state=42), random_state=42) #Validation F1 0.775 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=5, random_state=42), random_state=42) #Validation F1 0.779 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=6, random_state=42), random_state=42) #Validation F1 0.780 at first.
ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=7, random_state=42), random_state=42) #Validation F1 0.787 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=8, random_state=42), random_state=42) #Validation F1 0.764 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=9, random_state=42), random_state=42) #Validation F1 0.774 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=10, random_state=42), random_state=42) #Validation F1 0.775 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=11, random_state=42), random_state=42) #Validation F1 0.770 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=12, random_state=42), random_state=42) #Validation F1 0.759 at first.
#ensembletree_model=sklensemble.AdaBoostClassifier(skltree.DecisionTreeClassifier(max_depth=13, random_state=42), random_state=42) #Validation F1 0.751 at first.
#ensembletree_hyper={'n_estimators':[2,3,4,5,6,7,8,9,10,50,100,200,400,800,1000]}
#ensembletree_hyper={'n_estimators':[50,100,150]}
#ensembletree_hyper={'n_estimators':[90,100,110]}
#ensembletree_hyper={'n_estimators':[95,100,105]}
ensembletree_hyper={'n_estimators':[95,96,97,98,99,100,101,102,103,104,105]}
ensembletree_scorer=sklmet.make_scorer(sklmet.f1_score, average='macro')
ensembletree_GSCV=sklmodsel.GridSearchCV(ensembletree_model, ensembletree_hyper, scoring=ensembletree_scorer, cv=sklmodsel.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
ensembletree_GSCV.fit(x_train, y_train)
best_ensembletree_model=ensembletree_GSCV.best_estimator_
best_ensembletree_valF1=ensembletree_GSCV.best_score_
best_ensembletree_model.fit(x_train, y_train)

#Benchmark model for the second task. Perform hierarchical clustering on the dataset and average the variances of the last five mergers.
final_dendrogram=scpyhierclus.linkage(x_final, 'ward')
final_dendrogram[-5:,2].mean()

#Perform hierarchical clustering on each of the six subsets and show the variances of the last five mergers in each case.
tall_final_dendrogram=scpyhierclus.linkage(x_tall_final, 'ward')
tall_final_dendrogram[-5:,2]
ball_final_dendrogram=scpyhierclus.linkage(x_ball_final, 'ward')
ball_final_dendrogram[-5:,2]
aml_final_dendrogram=scpyhierclus.linkage(x_aml_final, 'ward')
aml_final_dendrogram[-5:,2]
nbl_final_dendrogram=scpyhierclus.linkage(x_nbl_final, 'ward')
nbl_final_dendrogram[-5:,2]
wt_final_dendrogram=scpyhierclus.linkage(x_wt_final, 'ward')
wt_final_dendrogram[-5:,2]
os_final_dendrogram=scpyhierclus.linkage(x_os_final, 'ward')
os_final_dendrogram[-5:,2]

#Dimensionality reduction by principal component analysis for the third task.
pca=skldecomp.PCA(n_components=11, random_state=42)
pca.fit(x_final)

#Visualise the results of the third task by plotting the information content in each component.
pca_outputs=pca.explained_variance_ratio_
plt.figure(figsize=(9,6))
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], pca_outputs, align='center')
plt.xlabel('Component Number')
plt.ylabel('Percentage of Variance Explained by a Component')
plt.title('Results of Principal Component Analysis')

#Test the classifiers using the testing data.
y_predict_tree=best_tree_model.predict(x_test)
f1_tree=sklmet.f1_score(y_test, y_predict_tree, average='macro')
f1_tree_details=sklmet.f1_score(y_test, y_predict_tree, average=None)
y_predict_nb=nb_model.predict(x_test)
f1_nb=sklmet.f1_score(y_test, y_predict_nb, average='macro')
f1_nb_details=sklmet.f1_score(y_test, y_predict_nb, average=None)
y_predict_svm_linear=best_svm_linear_model.predict(x_test)
f1_svm_linear=sklmet.f1_score(y_test, y_predict_svm_linear, average='macro')
f1_svm_linear_details=sklmet.f1_score(y_test, y_predict_svm_linear, average=None)
y_predict_svm_poly=best_svm_poly_model.predict(x_test)
f1_svm_poly=sklmet.f1_score(y_test, y_predict_svm_poly, average='macro')
f1_svm_poly_details=sklmet.f1_score(y_test, y_predict_svm_poly, average=None)
y_predict_svm_rbf=best_svm_rbf_model.predict(x_test)
f1_svm_rbf=sklmet.f1_score(y_test, y_predict_svm_rbf, average='macro')
f1_svm_rbf_details=sklmet.f1_score(y_test, y_predict_svm_rbf, average=None)
y_predict_mlp=mlp_model.predict_classes(np.array(x_test))
f1_mlp=sklmet.f1_score(y_test, y_predict_mlp, average='macro')
f1_mlp_details=sklmet.f1_score(y_test, y_predict_mlp, average=None)
y_predict_ensembletree=best_ensembletree_model.predict(x_test)
f1_ensembletree=sklmet.f1_score(y_test, y_predict_ensembletree, average='macro')
f1_ensembletree_details=sklmet.f1_score(y_test, y_predict_ensembletree, average=None)

#Test the benchmark model for the first task.
y_predict_benchmark=benchmark_model.predict(x_test)
f1_benchmark=sklmet.f1_score(y_test, y_predict_benchmark, average='macro')
f1_benchmark_details=sklmet.f1_score(y_test, y_predict_benchmark, average=None)

#Visualisation for the second task: dendrograms.
#scpyhierclus.dendrogram(final_dendrogram, truncate_mode='lastp', p=6, leaf_rotation=90., leaf_font_size=12., show_leaf_counts=True)
#scpyhierclus.dendrogram(tall_final_dendrogram)
#scpyhierclus.dendrogram(ball_final_dendrogram)
#scpyhierclus.dendrogram(aml_final_dendrogram)
#scpyhierclus.dendrogram(nbl_final_dendrogram)
#scpyhierclus.dendrogram(wt_final_dendrogram)
#scpyhierclus.dendrogram(os_final_dendrogram)
plt.xlabel('Sample')
plt.ylabel('Variance')
plt.title('Dendrogram for the B-ALL Subset')
scpyhierclus.dendrogram(ball_final_dendrogram, color_threshold=5.65, leaf_rotation=90., leaf_font_size=12., no_labels=True)
plt.xlabel('Sample')
plt.ylabel('Variance')
plt.title('Dendrogram for the AML Subset')
scpyhierclus.dendrogram(aml_final_dendrogram, color_threshold=5.65, leaf_rotation=90., leaf_font_size=12., no_labels=True)
plt.xlabel('Sample')
plt.ylabel('Variance')
plt.title('Dendrogram for the NBL Subset')
scpyhierclus.dendrogram(nbl_final_dendrogram, color_threshold=5.65, leaf_rotation=90., leaf_font_size=12., no_labels=True)
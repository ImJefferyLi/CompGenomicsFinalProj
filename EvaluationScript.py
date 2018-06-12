import pandas
import numpy
import matplotlib.pyplot as plt
import sys

PCA_data = pandas.read_csv(sys.argv[2] + "/ReducedData.csv")
PCA_data = PCA_data.as_matrix()

class_labels = pandas.read_csv(sys.argv[2] + "/classLabels.csv")
class_labels = class_labels.as_matrix()[:,0]

stage_labels = pandas.read_csv(sys.argv[2] + "/stageLabels.csv")
stage_labels = stage_labels.as_matrix()[:,0]

figurenum = 1

from sklearn import mixture
bics = []


for i in range(1,31): #determine the number of clusters from 1 to 30
	g = mixture.GMM(n_components=i, covariance_type = 'full', random_state = 1)
	g.fit(PCA_data)
	currBic = g.bic(PCA_data) #get the BIC of the current # of clusters
	bics.append(currBic) #track those BICs in a list

###START PLOT SNIPPET###
plt.figure(figurenum) #plot BIC vs # of clusters
figurenum = figurenum + 1
plt.plot(range(1,31), bics)
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('BIC Score Dependence on Number of Clusters Using GMM')
plt.savefig(sys.argv[2] + '/BICscore_alldata.png')
###END PLOT SNIPPET###

###START PLOT SNIPPET###
plt.figure(figurenum) #scatter plot of PC2 vs PC1
figurenum = figurenum + 1
plt.plot(PCA_data[0:100,0], PCA_data[100:200,1], 'b.', label = 'LUAD')
plt.plot(PCA_data[100:200,0], PCA_data[100:200,1], 'r.', label = 'LUSC')
plt.legend(loc = 'lower right', numpoints = 1)
plt.title('Scatter Plot of PC2 vs PC1 with Clinical Classification Labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_clinical_scatter.png')
###END PLOT SNIPPET###

###UNSUPERVISED PARTS
##first, try GMM, Kmeans, and HAC on 2 clusters (expected based on clinical classification)
import matplotlib.cm as cm

GMM = mixture.GMM(2,covariance_type = 'full')
biclusters1 = GMM.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(biclusters1)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(biclusters1) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, GMM (2 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_GMM_twoClust_scatter.png')
###END PLOT SNIPPET###

from sklearn import cluster

kms = cluster.KMeans(2)
biclusters2 = kms.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(biclusters2)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(biclusters2) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, K-means (2 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_Kmeans_twoClust_scatter.png')
###END PLOT SNIPPET###

aclu = cluster.AgglomerativeClustering(2)
biclusters3 = aclu.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(biclusters3)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(biclusters3) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, Agglomerative Clustering (2 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_HAC_twoClust_scatter.png')
###END PLOT SNIPPET###

from sklearn import metrics
from sklearn.metrics import pairwise_distances

#get their silhouette scores as a means of evaluating the unsupervised portion
bicluster1_score=metrics.silhouette_score(PCA_data, biclusters1, metric = 'euclidean')
bicluster2_score=metrics.silhouette_score(PCA_data, biclusters2, metric = 'euclidean')
bicluster3_score=metrics.silhouette_score(PCA_data, biclusters3, metric = 'euclidean')

#try GMM, Kmeans, and HAC on 7 clusters (expected based on BIC scores)
GMM = mixture.GMM(7,covariance_type = 'full')
nonclusters1 = GMM.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(nonclusters1)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(nonclusters1) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, GMM (7 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_GMM_sevenClust_scatter.png')
###END PLOT SNIPPET###

kms = cluster.KMeans(7)
nonclusters2 = kms.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(nonclusters2)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(nonclusters2) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, K-Means (7 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_Kmeans_sevenClust_scatter.png')
###END PLOT SNIPPET###

aclu = cluster.AgglomerativeClustering(7)
nonclusters3 = aclu.fit_predict(PCA_data)

###START PLOT SNIPPET###
clusters_unique = set(nonclusters3)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(nonclusters3) if x == currentCluster]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, Agglomerative Clustering (7 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_HAC_sevenClust_scatter.png')
###END PLOT SNIPPET###

#get their silhouette scores as a means of evaluating the unsupervised portion
noncluster1_score=metrics.silhouette_score(PCA_data, nonclusters1, metric = 'euclidean')
noncluster2_score=metrics.silhouette_score(PCA_data, nonclusters2, metric = 'euclidean')
noncluster3_score=metrics.silhouette_score(PCA_data, nonclusters3, metric = 'euclidean')

###START PLOT SNIPPET###
plt.figure(figurenum)
figurenum = figurenum + 1
n_groups = 3
twoclust = [bicluster1_score, bicluster2_score, bicluster3_score]
sevenclust = [noncluster1_score, noncluster2_score, noncluster3_score]
index = numpy.arange(n_groups)
bar_width = .35
r1 = plt.bar(index, twoclust, bar_width, color = 'b', label = 'Two Clusters')
r2 = plt.bar(index+bar_width, sevenclust, bar_width, color = 'r', label = 'Seven Clusters')
plt.xlabel('Unsupervised method')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score on Various Clustering Methods and Cluster Sizes')
plt.xticks(index+bar_width, ('GMM', 'Kmeans', 'HAC'))
plt.legend()
plt.savefig(sys.argv[2] + '/silhouette_barplot.png')
###END PLOT SNIPPET###

###SUPERVISED PORTION

from sklearn import ensemble
from sklearn import cross_validation
##For each of random forest, naive bayes, logistic regression, run 10 fold CV and get the average score (mean accuracy)
randfor = ensemble.RandomForestClassifier(100)
randforscores = cross_validation.cross_val_score(randfor, PCA_data, class_labels, cv = 10)

avgrandforscore = numpy.mean(randforscores)

from sklearn import naive_bayes
nbayes = naive_bayes.GaussianNB()
nbayesscores = cross_validation.cross_val_score(nbayes, PCA_data, class_labels, cv = 10)

avgnbayesscores = numpy.mean(nbayesscores)

from sklearn import linear_model
logreg = linear_model.LogisticRegression()
logregscores = cross_validation.cross_val_score(logreg, PCA_data, class_labels, cv = 10)

avglogregscores = numpy.mean(logregscores)

###START PLOT SNIPPET###
plt.figure(figurenum)
figurenum = figurenum + 1
supervisedscores = [avgrandforscore, avgnbayesscores, avglogregscores]
n_groups = 3
index = numpy.arange(n_groups)
bar_width = .35
plt.bar(index, supervisedscores, bar_width, color = 'b')
plt.xlabel('Supervised method')
plt.ylabel('Mean accuracy (10 fold CV)')
plt.title('Mean Accuracy of Classifying LUAD vs LUSC')
plt.xticks(index+bar_width, ('Random Forest', 'Naive Bayes', 'Logistic Regression'))
plt.savefig(sys.argv[2] + '/classification_accuracy_barplot.png')
###END PLOT SNIPPET###

###UNSUPERVISED DEEP DIVE
##Instead of having the two subtypes of NSCLC mixed, try clustering them separately to see how many clusters they each generate
my_dataset = pandas.read_csv(sys.argv[2] + "/FullDataSet.csv", index_col = 0) #read in the data - we added this step in later, without the foresight of realizing that by saving only the PCA-reduced data from our training script, we limited our ability to do this analysis.  So, we have to read in the full dataset again here.
my_dataset_reduced_ad = my_dataset.ix[0:100,0:20531]
my_dataset_reduced_sq = my_dataset.ix[100:200,0:20531]
my_matrix_ad = my_dataset_reduced_ad.as_matrix()
my_matrix_ad = numpy.log10(my_matrix_ad+1)
my_matrix_sq = my_dataset_reduced_sq.as_matrix()
my_matrix_sq = numpy.log10(my_matrix_sq+1)
from sklearn.decomposition import PCA
pca = PCA(50)
PCA_data_ad = pca.fit_transform(my_matrix_ad)

bics = []
for i in range(1,31):
	g = mixture.GMM(n_components=i, covariance_type = 'full', random_state = 1)
	g.fit(PCA_data_ad)
	currBic = g.bic(PCA_data_ad)
	bics.append(currBic)

###START PLOT SNIPPET###
plt.figure(figurenum)
figurenum = figurenum + 1
plt.plot(range(1,31),bics)
plt.xlabel('Number of clusters for Adenocarcinoma')
plt.ylabel('BIC')
plt.title('BIC Score Using GMM, Adenocarcinoma Only')
plt.savefig(sys.argv[2] + '/BIC_ad.png')
###END PLOT SNIPPET###

aclu = cluster.AgglomerativeClustering(5)
nonclusters3 = aclu.fit_predict(PCA_data_ad)

###START PLOT SNIPPET###
clusters_unique = set(nonclusters3)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(nonclusters3) if x == currentCluster]
	currentPCA = PCA_data_ad[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, AD only, Agglomerative Clustering (5 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_ad_HAC_scatter.png')
###END PLOT SNIPPET###

PCA_data_sq = pca.fit_transform(my_matrix_sq)
bics = []
for i in range(1,31):
	g = mixture.GMM(n_components=i, covariance_type = 'full', random_state = 1)
	g.fit(PCA_data_sq)
	currBic = g.bic(PCA_data_sq)
	bics.append(currBic)

###START PLOT SNIPPET###
plt.figure(figurenum)
figurenum = figurenum + 1
plt.plot(range(1,31),bics)
plt.xlabel('Number of clusters for Squamous Cell')
plt.ylabel('BIC')
plt.title('BIC Score Using GMM, Squamous Cell Only')
plt.savefig(sys.argv[2] + '/BIC_sq.png')
###END PLOT SNIPPET###

aclu = cluster.AgglomerativeClustering(4)
nonclusters3 = aclu.fit_predict(PCA_data_sq)

###START PLOT SNIPPET###
clusters_unique = set(nonclusters3)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(clusters_unique))))
for currentCluster in clusters_unique:
	indexes = [i for i, x in enumerate(nonclusters3) if x == currentCluster]
	currentPCA = PCA_data_sq[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentCluster)
plt.title('Scatter Plot of PC2 vs PC1, SQ only, Agglomerative Clustering (4 Clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_sq_HAC_scatter.png')
###END PLOT SNIPPET###

###Clinical associations

###START PLOT SNIPPET###
stage_labels = stage_labels.tolist()
stage_labels_unique = set(stage_labels)
plt.figure(figurenum)
figurenum = figurenum + 1
colors = iter(cm.rainbow(numpy.linspace(0,1,len(stage_labels_unique))))
for currentStage in stage_labels_unique:
	indexes = [i for i, x in enumerate(stage_labels) if x == currentStage]
	currentPCA = PCA_data[indexes,:]
	plt.plot(currentPCA[:,0], currentPCA[:,1], '.', color = next(colors), label = currentStage)
plt.legend(numpoints = 1, fontsize = 10)
plt.title('Scatter Plot of PC2 vs PC1, Classified by Tumor Stage')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(sys.argv[2] + '/PC2_PC1_clinical_stage_scatter.png')
###END PLOT SNIPPET###

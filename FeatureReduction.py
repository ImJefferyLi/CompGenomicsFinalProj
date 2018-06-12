import pandas
import numpy
figurenum = 1
import matplotlib.pyplot as plt
import sys

my_dataset = pandas.read_csv(sys.argv[2] + "/FullDataSet.csv", index_col = 0) #read in the data
my_dataset_reduced = my_dataset.ix[:,0:20531] #reduce to only numeric data
my_variances = []
for i in range(0,20531):
	currentGene = my_dataset_reduced.ix[:,i]
	currentVar = numpy.log(numpy.var(currentGene))
	my_variances.append(currentVar)
plt.figure(figurenum)
figurenum = figurenum + 1
plt.hist(my_variances, bins = range(-10, 26))
plt.title("Histogram of Gene Variances")
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.savefig(sys.argv[2] + '/variance_histogram.png')
class_labels = my_dataset.ix[:,20531].as_matrix() #save the class labels separately
stage_labels = my_dataset.ix[:,20532].as_matrix() #save the stage labels separately
my_matrix = my_dataset_reduced.as_matrix() #convert from dataframe to matrix
my_matrix = numpy.log10(my_matrix+1) #log(x+1) transform the data
from sklearn.decomposition import PCA
pca = PCA() #create PCA object
pca.fit(my_matrix) #fit the data to PCA, using max dimensions


top50 = pca.explained_variance_ratio_[0:50] #plot the contribution of each of the top 50 PC's

top50_cumsum = numpy.cumsum(top50) #get the cumulative sum of each of the top 50 PC's

###START PLOT SNIPPET###
plt.figure(figurenum)
figurenum = figurenum + 1
line1, = plt.plot(range(1,51),top50, 'b', label = "Variance per PC") #scree plot
line2, = plt.plot(range(1,51),top50_cumsum, 'r', label = "Cumulative variance") #cumulative sum on the same scree plot axis


plt.xlabel('Principal Component Number')
plt.ylabel('Percentage Variance')
plt.legend(loc = 'upper left', handles=[line1, line2])
plt.title("Scree Plot of Top 50 PC's")
plt.savefig(sys.argv[2] + '/top50_scree.png')
###END PLOT SNIPPET###

pca2 = PCA(50)
PCA_data = pca2.fit_transform(my_matrix) #apply dimensionality reduction, keep top 50 PCs
pandas.DataFrame(PCA_data).to_csv(sys.argv[2] + "/ReducedData.csv", index = False)
pandas.DataFrame(class_labels).to_csv(sys.argv[2] + "/classLabels.csv", index = False)
pandas.DataFrame(stage_labels).to_csv(sys.argv[2] + "/stageLabels.csv", index = False)
####MAYBE MOVE BELOW TO A DIFFERENT FILE?




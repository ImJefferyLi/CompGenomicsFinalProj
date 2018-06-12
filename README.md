# CompGenomicsFinalProj
Computational Genomics final project, spring 2016

This work was done for Computational Genomics at Johns Hopkins in the Spring term of 2016.
Jeffery Li and Dylan Hirch partnered on the project and received equal credit.
Raw data is not included in the github upload due to file sizes, but are made available for download in the links below.

---FOREWORD---
This README file will explain the separate components of the code at a high level, and give some overview of the data used.  In our submission, we have included the directories Data and Results.

IMPORTANT: If using a bash script to run the code, please make sure the data directory is stored as "Data" and NOT "Data/", else the code will not work.  Similarly, make sure the results directory is stored as "Results" and NOT "Results/".

Thus to run, use as follows:
sh process_data.sh Data Results
sh train_model.sh Data Results
sh test_model.sh Data Results

A large series of figures are generated in the Results directory.  While this may seem overwhelming, we hope that we were able to present them in a logical and appropriate manner in the formal writeup.

We include the results of the training portion of our project in our original submission.  Thus, you could theoretically run "testing_script.sh" on its own with no problem, should that be necessary.
---END FOREWORD---

---GLOSSARY---
LUAD = AD = Adenocarcinoma
LUSC = SC = Squamous cell carcinoma
---END GLOSSARY---

---RAW DATA AND PRE-PREPROCESSING---
Note: This section is only relevant if you are downloading the raw data and unzipping it.  If not doing this step, then feel free to skip this section.  We do not recommend starting from the zipped data - most of the data is unused, and we included the raw data we did use in our project submission.

The download link for our data is as follows:
For LUAD:
https://drive.google.com/open?id=0BxOU5YQp5apccE03eDRBaU1XUGs
For LUSC: 
https://drive.google.com/open?id=0BxOU5YQp5apcWGZtRmotZ0w4WVE

We provide the absolute raw data in its zipped format, named as the following:
1cdebf0b-3bc0-4f6b-aa55-c1f650a89769_LUAD.tar
2216aa5c-c837-4193-971b-4998be691942_LUSC.tar
Feel free to unzip and double check these as needed, though we've also submitted the relevant data files extracted from the tar files (with the unnecessary files removed), so unzipping the data yourself is not necessary.

The directory Data/LUAD_raw contains the relevant data files associated with adenocarcinoma
The directory Data/LUSC_raw contains the relevant data files associated with squamous cell
The directory Data/Clinical contains the relevant data files associated with both adenocarcinoma and squamous cell, but only for patient information

We had to perform some minor pre-preprocessing that did not use any sort of scripting, so there is no code associated with the following steps:
	-The LUAD and LUSC filesets provide 6 types of data files, and we were only interested in the ones that ended in "genes.normalized_results", thus, in the LUAD_raw and LUSC_raw folders, we only kept these data files and removed all the other data that comes out of unzipping the tar files.
	-The clinical filesets also provided a large dump of clinical data - we are only interested in patient diagnostic data, so we onl kept the *_clinical_patient_* files in the Data/Clinical folder.
		-NO FURTHER EDITING ON THESE RAW DATA WERE PERFORMED IN PRE-PREPROCESSING.
	-Both the LUAD and LUSC filesets have an associated file called file_manifest.txt.  This file does not contain any raw data but rather maps patient barcodes to file names.  We use this data for our preprocessing.  We performed the following pre-preprocessing on these files:
		-For the column named "File Name", changed to "FileName" so all columns would be whitespace delimited
		-Manually removed every row that was not type "RNASeq" using a text editor
		-Because the whitespace delimiter was inconsistent across columns, we opened the file in Excel and saved as csv, so all columns are now comma delimited
			-These files are renamed "file_manifest_LUAD.csv", and "file_manifest_LUSC.csv", and are stored in the Data directory
	
---END SECTION ON RAW DATA AND PRE-PREPROCESSING---

---DATA PREPROCESSING---
	-The R script MainDatasetBuilder.R will combine all the LUAD and LUSC data into a single matrix, saved as a .csv file called "FullDataSet.csv"
	-The rows are the patients, the columns are the genes (features), and each row is marked as either LUAD or LUSC.  The pathological stage of the tumor is also joined to this data set.
	-The output is saved as FullDataSet.csv in the Results directory.
	-You can run this file as 'Rscript MainDatasetBuilder.R Data Results', or alternatively use the provided bash script and run as 'sh process_data.sh Data Results'
---END DATA PREPROCESSING---

---MODEL TRAINING---
	-Our project did not revolve around building a single complex machine learning algorithm and then tuning hyperparemeters to get a single set of results.  Rather, we combined a series of simple free-library methods as a means of discovering something interesting in our data.  Thus, our "model training" portion is a bit different from traditional means
	-The python script FeatureReduction.py will read in the preprocessed data and explore some of the dimensionality constraints.  It will generate a histogram plot of the variances in each gene, and save it to the Results directory.  It will also generate a scree plot after performing PCA and keeping the top 50 principal components and save it to the Results directory.  Finally, it will transform the data along the top 50 PC's and save the reduced data to the Results directory, along with associated type (LUAD vs LUSC) and clinical stage.
	-You can run this file as 'python FeatureReduction.py Data Results', or alternatively use the provided bash script and run as 'sh train_model.sh Data Results'
---END MODEL TRAINING---

---MODEL TESTING---
	-Like in model training, this project does not follow the traditional train/test paradigm.
	-For each machine learning method we employ, training and testing are all very intricately linked by the sklearn library.  Thus, a lot of "training" also happens in this script, though the models are simple, the data is relatively small, and thus the code runs quickly.
	-In this script, we run both unsupervised and supervised algorithms using the sklearn library.  We generate a battery of plots that are all output to the Results directory.
		-Please note: Due to some random initialization with various methods, the plots that are generated on every run may be different.
	-You can run this file as 'python EvaluationScript.py Data Results', or alternatively use the provided bash script and run as 'sh test_model.sh Data Results'
---END MODEL TESTING---

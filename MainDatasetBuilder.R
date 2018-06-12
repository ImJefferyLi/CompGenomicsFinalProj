#TCGA gives data for each patient in a separate file.  This script will combine all files into a single file for use for the remainder of the pipeline

args = commandArgs(trailingOnly = TRUE)

LUAD_manifest <- read.csv(paste(args[1], "/file_manifest_LUAD.csv", sep = ""), stringsAsFactors = F)
LUSC_manifest <- read.csv(paste(args[1], "/file_manifest_LUSC.csv", sep = ""), stringsAsFactors = F)

LUAD_rowhits <- grep("genes.normalized_results", LUAD_manifest[,"FileName"]) #the manifest has all 6 filetypes, only want genes.normalized_results.  Get the row numbers of those files.
LUSC_rowhits <- grep("genes.normalized_results", LUSC_manifest[,"FileName"])

LUAD_relevantFiles <- LUAD_manifest[LUAD_rowhits,] #get the filenames using the row numbers found above
LUSC_relevantFiles <- LUSC_manifest[LUSC_rowhits,]

for (i in 1:nrow(LUAD_relevantFiles)) {
  currentFileName <- LUAD_relevantFiles[i,"FileName"] #look at the files one by one
  currentFileName <- paste(args[1], "/LUAD_raw/", currentFileName, sep = "") #append the directory structure to the front
  mydata <- read.table(currentFileName, header = T, stringsAsFactors = F) #read in the data
  if (i == 1) { #first read instance
    LUAD_dataset <- mydata 
    colnames(LUAD_dataset)[2] <- LUAD_relevantFiles[i,"Sample"] #store the name of the TCGA patient
  }
  else {
    LUAD_dataset <- cbind(LUAD_dataset, mydata[,2]) #append the current patient to the growing data structure
    colnames(LUAD_dataset)[ncol(LUAD_dataset)] <- LUAD_relevantFiles[i,"Sample"] #store the name of the TCGA patient
  }
}
rownames(LUAD_dataset) <- LUAD_dataset[,1] #set the rownames to be the gene names
LUAD_dataset <- LUAD_dataset[,2:ncol(LUAD_dataset)] #remove the first column, which was a column of gene names, so now all matrix data is numeric
LUAD_dataset <- t(LUAD_dataset) #transpose the matrix, so rows are samples and columns are genes (features)
Type <- rep("LUAD", times = nrow(LUAD_dataset)) #label the rows to be LUAD
LUAD_dataset <- cbind(LUAD_dataset, Type) #append the row labels

###REPEAT FOR LUSC DATA

for (i in 1:nrow(LUSC_relevantFiles)) {
  currentFileName <- LUSC_relevantFiles[i, "FileName"]
  currentFileName <- paste(args[1], "/LUSC_raw/", currentFileName, sep = "")
  mydata <- read.table(currentFileName, header = T, stringsAsFactors = F)
  if (i == 1) {
    LUSC_dataset <- mydata
    colnames(LUSC_dataset)[2] <- LUSC_relevantFiles[i, "Sample"]
  }
  else {
    LUSC_dataset <- cbind(LUSC_dataset, mydata[,2])
    colnames(LUSC_dataset)[ncol(LUSC_dataset)] <- LUSC_relevantFiles[i, "Sample"]
  }
}
rownames(LUSC_dataset) <- LUSC_dataset[,1]
LUSC_dataset <- LUSC_dataset[,2:ncol(LUSC_dataset)]
LUSC_dataset <- t(LUSC_dataset)
Type <- rep("LUSC", times = nrow(LUSC_dataset))
LUSC_dataset <- cbind(LUSC_dataset, Type)

##COMBINE INTO ONE

FullDataSet <- rbind(LUAD_dataset, LUSC_dataset)
LUAD_clinical <- read.table(paste(args[1], "/Clinical/nationwidechildrens.org_clinical_patient_luad.txt", sep = ""), stringsAsFactors = F, header = T, row.names = 1, sep = "\t") #read in LUAD clinical
LUSC_clinical <- read.table(paste(args[1], "/Clinical/nationwidechildrens.org_clinical_patient_lusc.txt", sep = ""), stringsAsFactors = F, header = T, row.names = 1, sep = "\t") #read in LUSC clinical

LUAD_stage <- LUAD_clinical[3:nrow(LUAD_clinical),c(1, 23)] #get only the TCGA barcode and stage data
LUSC_stage <- LUSC_clinical[3:nrow(LUSC_clinical),c(1, 22)]

for (i in 1:nrow(LUAD_stage)) {
  LUAD_stage[i,1] <- paste(LUAD_stage[i,1], "01", sep = "-") #reformat so the TCGA barcodes match the RNA-seq TCGA barcodes
}

for (i in 1:nrow(LUSC_stage)) {
  LUSC_stage[i,1] <- paste(LUSC_stage[i,1], "01", sep = "-")
}

all_stage <- rbind(LUAD_stage, LUSC_stage) #combine the LUAD and LUSC clinical data
order_vec <- 1:nrow(FullDataSet) #keep the order before the later join
FullDataSet <- cbind(order_vec, FullDataSet)
FullDataSet <- cbind(FullDataSet, rownames(FullDataSet)) #append a column using the TCGA barcode for the dataframe join
colnames(FullDataSet)[ncol(FullDataSet)] <- "TCGA_barcode"
colnames(all_stage)[1] <- "TCGA_barcode"
fulldata_plusclinical <- merge(FullDataSet, all_stage, by = "TCGA_barcode", all = T) #join the dataframes using the TCGA barcode
rownames(fulldata_plusclinical) <- fulldata_plusclinical[,1]
fulldata_plusclinical <- fulldata_plusclinical[order(as.numeric(as.character(fulldata_plusclinical[,2]))),] #restore the original order, ie. all LUAD first, then all LUSC

fulldata_plusclinical <- fulldata_plusclinical[,3:ncol(fulldata_plusclinical)]
write.csv(fulldata_plusclinical, file = paste(args[2], "/FullDataSet.csv", sep = ""))



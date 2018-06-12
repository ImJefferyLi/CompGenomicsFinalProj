#!bin/bash

echo "Starting preprocessing of data..."
Rscript MainDatasetBuilder.R $1 $2
echo "...done"

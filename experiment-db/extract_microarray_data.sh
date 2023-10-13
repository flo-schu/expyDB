#!bin/bash

output="data/processed_data/ds1_microarray_data"
mkdir $output
tar -xvf data/dataset_1/GSE109496_RAW.tar -C $output

for file in $output/*.gz; do
    gzip -d "$file"
done

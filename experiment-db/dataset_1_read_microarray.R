library(toxprofileR2)
library(tidyverse)
# substance <- "diuron"
# targetfile = paste0("./data/dataset_1/targetsfile_", substance, ".csv")

targetfile = paste0("./data/dataset_1/targetsfile.csv")
datadir = paste0("./data/processed_data/ds1_microarray_data")

data <- toxprofileR2::import_array_data(targetfile = targetfile,
    datadir = datadir, output = T, removeOutliers = T, qc_coeff = c(ks = 3, 
        sum = 3, iqr = 3, q = 3, d = 1), qc_sum = 1)

    targets <- limma::readTargets(file = targetfile, sep = ",", row.names=TRUE)

single_target <- targets[1,]

toxprofileR2::normalise_batch()
raw <- limma::read.maimages(
    files = single_target,
    source = "agilent",
    path = datadir,
    names = single_target$names,
    green.only = T,
    columns = list(
    E = "gProcessedSignal",
    Processederror = "gProcessedSigError",
    Median = "gMedianSignal",
    Medianb = "gBGMedianSignal",
    processedSignal = "gProcessedSignal",
    isNonUniform = "gIsFeatNonUnifOL",
    isNonUniformBG = "gIsBGNonUnifOL",
    isPopOutlier = "gIsFeatPopnOL",
    isPopOutlierBG = "gIsBGPopnOL",
    manualFlag = "IsManualFlag",
    posandsigf = "gIsPosAndSignif",
    aboveBG = "gIsWellAboveBG",
    bgSubSignal = "gBGSubSignal"
    ),
    verbose = T
)

library(archive)
library(readr)


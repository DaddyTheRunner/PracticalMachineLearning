## This script contains the helper functions used in
## the Practical Machine Learning project

## Load required pacakges
require(dplyr)
require(caret)

## The following function loads the pml data
load.pml.data <- function (seed=1234) {
  ## Set the RNG seed
  set.seed(seed)
  
  grading.ds <- read.csv("pml-testing.csv",
                         na.strings = c("NA","","#DIV/0!"))
  train.ds <- read.csv("pml-training.csv",
                       na.strings = c("NA","","#DIV/0!"))
  
  ## Convert to tables
  grading.ds <- tbl_df(grading.ds)
  train.ds <- tbl_df(train.ds)
  
  ## Remove the first seven book keeping columns from both data sets
  grading.ds <- grading.ds[,-(1:7)]
  train.ds <- train.ds[,-(1:7)]
  
  ## Find the columns with NAs in the grading set and remove them
  ## from both sets
  na.cols <- is.na(grading.ds[1,])
  grading.ds <<- grading.ds[,!na.cols]
  train.ds <- train.ds[,!na.cols]
  
  ## Split the training set into a training set and a validation set
  inTraining <- createDataPartition(train.ds$classe, p = 0.75, list = FALSE)
  training.ds <<- train.ds[inTraining,]
  validation.ds <<- train.ds[-inTraining,]
}


## The following function calculates the relative importance of
## all of the features when building a Random Forrest model
selectFeatures <- function (data, sampleSize = 0.1, seed = 1234,
                            repeats = 10) {
  ## Set the random seed
  set.seed(seed)
  
  ## Set up the control parameters
  fitControl <- trainControl(method="oob", returnData=FALSE)
  
  imp.vars <- list(0)
  
  for (i in 1:repeats) {
    ## Resample the data
    inSample <- createDataPartition(data$classe, p = sampleSize, list = FALSE)
    sample.ds <- data[inSample,]
    cat("\nStarting the model training at:  "); print(Sys.time())
    model <- train(classe ~ ., data=sample.ds, method="rf",
                   trControl=fitControl, verbose=FALSE, model=FALSE)
    cat("\nFinished the model training at:  "); print(Sys.time())
    imp.vars[[i]] <- varImp(model)$importance
  }
  
  return(imp.vars)
}


## The following function calculates the average importance for
## each varialbe over each of the runs
avgImp <- function (varImp) {
  result <- varImp[[1]]
  for (i in 2:length(varImp)) { result <- result + varImp[[i]] }
  result <- result / length(varImp)
  
  return(result)
}


## The following function writes the answers out to individual
## text files
## It was provided by the instructors
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

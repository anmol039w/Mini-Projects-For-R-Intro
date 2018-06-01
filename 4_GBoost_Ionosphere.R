#======================================================
# Datase:  ionosphere dataset
#
# 
# Problem: Determine if a given signal is Good (g) or Bad (b).
# Good RADAR returns a nice structure, BAD RADAR do not return good
# structures.
#
# Example Attributes: Angle of the Sun, Ionization storms (0/1), 
#                     
# 
# This is a dataset available from the UCI Machine Learning
# Repository. This dataset describes high-frequency antenna 
# returns from high energy particles in the atmosphere and whether
# the return shows structure or not. 
# The problem is a binary classification that 
#   contains 351 instances and 35 numerical attributes.
#
# 
# Source of Data:
# ----------------
# Space Physics Group
# Applied Physics Laboratory
# Johns Hopkins University
# Johns Hopkins Road
# Laurel, MD 20723
#
# 
#======================================================

# Load libraries
library(mlbench) # ML benchmark problems with UCI repository, for dataset
library(caret) # Classification and Regresssion training
library(caretEnsemble) # For ensembles of caret models

# Load the dataset
data(Ionosphere)
dataset <- Ionosphere
dataset <- dataset[,-2]
dataset$V1 <- as.numeric(as.character(dataset$V1))

# Note that the first attribute was a factor (0,1) and has been transformed to be numeric for consistency with all of the other numeric attributes. 
# Also note that the second attribute is a constant and has been removed.

head(dataset)

# Example of Boosting Algorithms
# The function trainControl generates parameters that further control how models are created, 
# with possible values:
# method - crossvalidation, k=10, do cross validation 3 times
control <- trainControl(method="cv", number=10, repeats=3) # Cross validation
seed <- 7
metric <- "Accuracy"

# C5.0
set.seed(seed)
fit.c50 <- train(Class~., data=dataset, method="C5.0", metric=metric, trControl=control)

# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Class~., data=dataset, method="gbm", metric=metric, trControl=control)

# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)


# We can see that the C5.0 algorithm produces a more accurate model with an accuracy of 94.58%.


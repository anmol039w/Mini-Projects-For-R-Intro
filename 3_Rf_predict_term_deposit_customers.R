#==================================================================================

# Build a predictive model which will identify customers who are more likely to 
# respond to term deposit cross sell campaign.

#----------------------------------------------------------------------------------
# Data Set: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/

# Input dataset has 20 independent variables and a target variable. 
# The target variable y is binary (Yes or No).
#----------------------------------------------------------------------------------
# A marketing department of a bank runs various marketing campaigns for cross-selling 
# products, improving customer retention and customer services.

# In this example, the bank wanted to cross-sell term deposit product to its customers.
# Contacting all customers is costly and does not create good customer experience.
#==================================================================================

library(randomForest)

## Read data
termCrosssell<-read.csv(file="bank-5000.csv",header = T, sep = ";")

## Explore data frame

names(termCrosssell)

##  [1] "age"            "job"            "marital"        "education"     
##  [5] "default"        "housing"        "loan"           "contact"       
##  [9] "month"          "day_of_week"    "duration"       "campaign"      
## [13] "pdays"          "previous"       "poutcome"       "emp.var.rate"  
## [17] "cons.price.idx" "cons.conf.idx"  "euribor3m"      "nr.employed"   
## [21] "y"

# How much % of target variable has "yes"...
table(termCrosssell$y)/nrow(termCrosssell)

## 
##        no       yes 
## 0.8873458 0.1126542

# termCrosssell$y

#---------------------------------------------------------------
# split the data sample into development(train) and validation(test) samples.
#---------------------------------------------------------------
sample.ind <- sample(2, 
                     nrow(termCrosssell),
                     replace = T)
#                     prob = c(0.6,0.4))

cross.sell.dev <- termCrosssell[sample.ind==1,] # 1st sample for train
cross.sell.val <- termCrosssell[sample.ind==2,] # 2nd sample for test/validation 


#---------------------------------------------------------------
# Check whether Both development and validation samples have similar target variable distribution.
#---------------------------------------------------------------

table(cross.sell.dev$y)/nrow(cross.sell.dev)
## 
##        no       yes 
## 0.8859289 0.1140711


table(cross.sell.val$y)/nrow(cross.sell.val)
## 
## no yes 
## 0.8894524 0.1105476

#---------------------------------------------------------------
# What is the type of the target variable
# if it is factor means it's a classification target
#---------------------------------------------------------------
class(cross.sell.dev$y)

## [1] "factor"

#---------------------------------------------------------------
# Make formaula of all the features
#---------------------------------------------------------------
varNames <- names(cross.sell.dev)
# Exclude ID or Response variable
varNames <- varNames[!varNames %in% c("y")]

# add + sign between exploratory variables
varNames1 <- paste(varNames, collapse = "+")

# Add response variable and convert to a formula object

rf.form <- as.formula(paste("y", varNames1, sep = " ~ "))

#---------------------------------------------------------------
# Building Random Forest using R
#---------------------------------------------------------------

# 500 decision trees or a forest has been built using the Random 
# Forest algorithm based learning. We can plot the error rate across decision trees. 
# The plot seems to indicate that after 100 decision trees, there is not a 
# significant reduction in error rate.


cross.sell.rf <- randomForest(rf.form,
                              cross.sell.dev,
                              importance=T)

#plot(cross.sell.rf)

#---------------------------------------------------------------
# Find out the important features
#---------------------------------------------------------------

# Variable importance plot is also a useful tool and can be plotted 
# using varImpPlot function. Top 5 variables are selected and plotted
# based on Model Accuracy and Gini value. We can also get a table with 
# decreasing order of importance based on a measure (1 for model accuracy 
# and 2 node impurity)

# Variable Importance Plot
varImpPlot(cross.sell.rf,
           sort = T,
           main="Variable Importance",
           n.var=5)

# Variable Importance Table
var.imp <- data.frame(importance(cross.sell.rf,type=2))
# make row names as columns
var.imp$Variables <- row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),]

# Based on Random Forest variable importance, the variables 
# could be selected for any other predictive modelling techniques or machine learning.

#---------------------------------------------------------------
# Measuer the accuracy of the RF model
#---------------------------------------------------------------

# Predicting response variable
cross.sell.dev$predicted.response <- predict(cross.sell.rf ,cross.sell.dev)

# Create confusion matrix
library(e1071)
library(caret)

## Loading required package: lattice
## Loading required package: ggplot2
# Create Confusion Matrix
confusionMatrix(data=cross.sell.dev$predicted.response,
                reference=cross.sell.dev$y)
#                positive='yes')

# Predicting response variable
cross.sell.val$predicted.response <- predict(cross.sell.rf ,cross.sell.val)

# Create Confusion Matrix
confusionMatrix(data=cross.sell.val$predicted.response,
                reference=cross.sell.val$y,
                positive='yes')


# Output of confusion matrix
#=============================================
# 1. Atable of TP, TN, FP and FN
# 2. Accuracy - TP + TN / all
#
#   Accuracy = (TP+TN)/(TP+FP+TN+FN)
#
# 3. 95% CI = 95% Confidence Interval = It means that if the
     #same population is sampled on numerous occasions and 
#    interval estimates are made on each occasion, the resulting
#    intervals would bracket the true population parameter in 
#    approximately 95 % of the cases. 
# 4. No Information rate - the proportion of classes that you would guess right 
#                          if you randomly allocated them. Acc must be > NIR
# 5. p-value - 1-specificity
# 6. kappa - Kappa = (observed accuracy - expected accuracy)/(1 - expected accuracy)
# 
# 7. Mcmnear Test P-value - Follows chi2 test - (TN-FP)**2/(TN+FP)
# 8. Sensitivity - When it is actually "yes" how often it predicts "Yes".
#      Recall = Sensitivity = TP/(TP+FN) = TP/ ( Actual Yes)
#
# 9. specificity - Specificity measures true negative rate. 
#         When it is actually "No", how often it is "No".
#         Specificity= TN/(TN+FP) = TN/(Actual No)
# 10. pos pred value
#         similar to precision, except that it takes prevalence into account. 
#         In the case where the classes are perfectly balanced (meaning 
#         the prevalence is 50%), the positive predictive value (PPV) 
#         is equivalent to precision. 
#
#         PPV = TP/TP+FP
# 11. Neg pred value
#         NPV = TN/TN+FN
# 12. prevalence
#              Prevalence= Actual yes/Total = ( FN+TP)/(TP+FP+TN+FN) 
#
# 13. Detection RAte
#              DR = TP/(ALL)
# 14. Detection prevalence
#              DP = TP + TN /ALL
# 15. Balanced accuracy
#               BA = sensitivity+specificity /2
# 16. positive class - Yes or NO
# 17. Misclassification Rate = 1 - Accuracy
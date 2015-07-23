# A prediction model to predict activity quality from activity monitors.
Author: Rainer-Anton Englisch

## Executive summary
The task of the course project can be summarized by a quote of the course projects summary:
"Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)."

Simply put: We want to build a prediction model to predict activity quality from activity monitors. 

## Predictor Analysis and Reduction 
Before letting caret to create a prediction model we will try to reduce the number of predictors in order to speed up the creation of the prediction model. First we load the training data set and remove the column "X"" to avoid overfitting by observation number.

The activity quality of an observation is classified by the factor variable classe which we store in seperate variables for later use for training and prediction.




```r
set.seed(1312)
pml_training = read.csv("pml-training.csv")
pml_training = pml_training[,-1]
dim(pml_training)
```

```
## [1] 19622   159
```


Next we split the original training in a new training and a test data. The test data will be used later for estimating the out of sample error.

The activity quality of an observation is classified by the factor variable classe which we store in seperate variables for later use for training and prediction.


```r
library(caret)
inTrain <- createDataPartition(pml_training$classe,p=0.75, list=FALSE)
training <- pml_training[inTrain,]
trainclasse <- training$classe
testing <- pml_training[-inTrain,]
testingclasse <- testing$classe
```

### Remove predictors with near zero variability
Next we throw out all predictors that have near zero variability because we assume
that these predictors have minimal influence on the prediction.


```r
nZV <- nearZeroVar(training,saveMetrics=TRUE)
# extract the predictor names which have near zero variability
nzvcolnames <- rownames(nZV[nZV$nzv==TRUE,])
# compute the index of these predictor  names in the training data frame
nzvcolindex <- which(names(training) %in% nzvcolnames)
# remove these predictors
training <- training[,-nzvcolindex]
# print these predictors
nzvcolnames
```

```
##  [1] "new_window"              "kurtosis_roll_belt"     
##  [3] "kurtosis_picth_belt"     "kurtosis_yaw_belt"      
##  [5] "skewness_roll_belt"      "skewness_roll_belt.1"   
##  [7] "skewness_yaw_belt"       "max_yaw_belt"           
##  [9] "min_yaw_belt"            "amplitude_yaw_belt"     
## [11] "avg_roll_arm"            "stddev_roll_arm"        
## [13] "var_roll_arm"            "avg_pitch_arm"          
## [15] "stddev_pitch_arm"        "var_pitch_arm"          
## [17] "avg_yaw_arm"             "stddev_yaw_arm"         
## [19] "var_yaw_arm"             "kurtosis_roll_arm"      
## [21] "kurtosis_picth_arm"      "kurtosis_yaw_arm"       
## [23] "skewness_roll_arm"       "skewness_pitch_arm"     
## [25] "skewness_yaw_arm"        "kurtosis_roll_dumbbell" 
## [27] "kurtosis_picth_dumbbell" "kurtosis_yaw_dumbbell"  
## [29] "skewness_roll_dumbbell"  "skewness_pitch_dumbbell"
## [31] "skewness_yaw_dumbbell"   "max_yaw_dumbbell"       
## [33] "min_yaw_dumbbell"        "amplitude_yaw_dumbbell" 
## [35] "kurtosis_roll_forearm"   "kurtosis_picth_forearm" 
## [37] "kurtosis_yaw_forearm"    "skewness_roll_forearm"  
## [39] "skewness_pitch_forearm"  "skewness_yaw_forearm"   
## [41] "max_roll_forearm"        "max_yaw_forearm"        
## [43] "min_roll_forearm"        "min_yaw_forearm"        
## [45] "amplitude_yaw_forearm"   "avg_roll_forearm"       
## [47] "stddev_roll_forearm"     "var_roll_forearm"       
## [49] "avg_pitch_forearm"       "stddev_pitch_forearm"   
## [51] "var_pitch_forearm"       "avg_yaw_forearm"        
## [53] "stddev_yaw_forearm"      "var_yaw_forearm"
```
We removed 54 predictors in the training data frame.

### Remove predictors with high linear correlation

Next we want to throw out predictors that have a high linear correlation.


```r
#As the correlation matix can only be computed for numeric variables we need to identify numeric variables
colsnumeric <- sapply(training, is.numeric)

#compute the correlation matrix
cortraining <- cor(training[,colsnumeric],use="pairwise.complete.obs")
```

Within the computed correlation matrix we select all predictors that have a high correlation. Let us define a high corelation as a value equal or greater than 0.7. Thus let's find these predictors and remove them from the training data set.

```r
# retrieve variables that have a correlation greater or equal to 0.7
highlyCor <- findCorrelation(cortraining, 0.70,verbose=FALSE)
highlyCorcolnames <- colnames(training)[highlyCor]
# remove the highly correlated predictors from the training data frame
training <- training[,-highlyCor]
# print the removed predictors
highlyCorcolnames
```

```
##  [1] "yaw_belt"                "yaw_dumbbell"           
##  [3] "stddev_pitch_belt"       "roll_belt"              
##  [5] "accel_belt_y"            "max_roll_belt"          
##  [7] "gyros_belt_y"            "stddev_pitch_dumbbell"  
##  [9] "accel_belt_x"            "amplitude_pitch_belt"   
## [11] "total_accel_belt"        "cvtd_timestamp"         
## [13] "num_window"              "accel_dumbbell_x"       
## [15] "max_picth_belt"          "pitch_belt"             
## [17] "roll_dumbbell"           "gyros_belt_z"           
## [19] "pitch_dumbbell"          "amplitude_roll_dumbbell"
## [21] "gyros_dumbbell_y"        "gyros_dumbbell_z"       
## [23] "min_roll_dumbbell"       "min_pitch_dumbbell"     
## [25] "stddev_roll_dumbbell"    "var_roll_dumbbell"      
## [27] "accel_dumbbell_y"        "var_accel_dumbbell"     
## [29] "var_pitch_dumbbell"      "min_pitch_forearm"      
## [31] "accel_dumbbell_z"        "user_name"              
## [33] "magnet_arm_z"            "gyros_arm_y"            
## [35] "accel_belt_z"            "total_accel_dumbbell"   
## [37] "amplitude_roll_arm"      "min_pitch_belt"         
## [39] "accel_arm_x"             "amplitude_roll_belt"    
## [41] "var_total_accel_belt"    "accel_arm_z"            
## [43] "gyros_forearm_z"         "var_roll_belt"          
## [45] "var_accel_arm"           "min_roll_belt"          
## [47] "var_pitch_belt"          "gyros_forearm_x"
```
We removed 48 highly correlated predictors.

### Remove predictors which are unimportant for prediction

Now we want to quickly create  a small prediction model in order to
query the importance of the variables for the prediction model.
Let's fit a model, print the important predictors and keep these predictors in the training data set.

```r
modFit <- train(classe ~.,data=training,method="rpart")
```

```
## Loading required package: rpart
```

```r
# print summary of the model fit
modFit
```

```
## CART 
## 
## 14718 samples
##    56 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 311, 311, 311, 311, 311, 311, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.06086957  0.4787784  0.3396471  0.04989614   0.06068268
##   0.09782609  0.4157876  0.2434827  0.04062212   0.05240168
##   0.20000000  0.3183308  0.1047526  0.08170700   0.09903919
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.06086957.
```

```r
# retrieve the variable importance list
importance <- varImp(modFit, scale=FALSE)
importance <- importance[[1]]
# bind the rownames as columns 
importance <- cbind(rownames(importance),importance )
# retrieve column names which have importance greater zero
impcolnames <- importance[importance$Overall>0.0,1]
# print important variables
impcolnames
```

```
##  [1] amplitude_yaw_arm avg_roll_belt     avg_roll_dumbbell
##  [4] magnet_belt_y     magnet_dumbbell_y min_roll_arm     
##  [7] pitch_forearm     roll_forearm      stddev_roll_belt 
## [10] stddev_yaw_belt   var_accel_forearm var_yaw_belt     
## 56 Levels: accel_arm_y accel_forearm_x accel_forearm_y ... yaw_forearm
```

```r
# retrieve indexes of column names
impcolindex <- which(names(training) %in% impcolnames)
removedCols <- ncol(training) - length(impcolnames)
# keep important columns in training data frame
training <- training[,impcolindex]
```

In the last step we have removed 45 unimportant predictors.

### Train final prediction model
Now that we have reduced the predictors significantly from 160 to 12 predictors we will fit a more complex machine learning algorithm based on random forest. Additionally we will use cross validation to minimize in order to get a out of sample error rate.


```r
fitControl <- trainControl(method = "repeatedcv",number = 5, repeats = 5)
preObj <- preProcess(training,method=c("knnImpute"))
trainingImputed <- predict(preObj,newdata=training)
modFit <- train(trainclasse ~.,data=trainingImputed,
                method="rf"
                #method="rpart"
              ,trControl = fitControl
              )
# print summary of model
print(modFit)
```

```
## Random Forest 
## 
## 14718 samples
##    11 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 5 times) 
## 
## Summary of sample sizes: 11774, 11772, 11775, 11775, 11776, 11774, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.8107227  0.7602558  0.007514918  0.009465495
##    7    0.8219469  0.7746403  0.006147833  0.007731184
##   12    0.8118910  0.7619735  0.006685442  0.008377148
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 7.
```

### Compute the out of sample error based on a seperate training (or validation) set
Let's use our serperate test (or validation) set to compute an out of sample error.


```r
# subselect the predictors used for training
traincolnames <- colnames(training)
traincolindex <- which(names(testing) %in% traincolnames)
testing <- testing[,traincolindex]
preObj <- preProcess(testing,method=c("knnImpute"))
testingImputed <- predict(preObj,newdata=testing)
 predictions <- predict(modFit,newdata=testingImputed)
# summarize results
confusionMatrix(predictions, testingclasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1325  155  185  123   41
##          B   24  630   61  114   82
##          C   12   46  503   26   60
##          D    7   25   11  482   14
##          E   27   93   95   59  704
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7431          
##                  95% CI : (0.7306, 0.7553)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6707          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9498   0.6639   0.5883  0.59950   0.7814
## Specificity            0.8564   0.9290   0.9644  0.98610   0.9316
## Pos Pred Value         0.7244   0.6915   0.7774  0.89425   0.7198
## Neg Pred Value         0.9772   0.9201   0.9173  0.92623   0.9498
## Prevalence             0.2845   0.1935   0.1743  0.16395   0.1837
## Detection Rate         0.2702   0.1285   0.1026  0.09829   0.1436
## Detection Prevalence   0.3730   0.1858   0.1319  0.10991   0.1994
## Balanced Accuracy      0.9031   0.7964   0.7764  0.79280   0.8565
```

```r
outOfSampleAccuracy <- sum(predictions==testingclasse)/length(testingclasse)
outOfSampleAccuracy
```

```
## [1] 0.7430669
```

```r
outOfSampleError <- 1-outOfSampleAccuracy
outOfSampleError
```

```
## [1] 0.2569331
```

### Compare in sample error and out of bag sample error and out of sample error
Finally we want to compare the in sample error and the out of sample errors.  Both errors need to be derived from accuracy variables  stored within the results variable within the prediction model.


```r
# the out of bag sample accuracy of the prediction model
inSampleError <- 1-max(modFit$results$Accuracy)
inSampleError
```

```
## [1] 0.1780531
```

```r
cvoutOfSampleError <- 1-max(modFit$results$Kappa)
cvoutOfSampleError
```

```
## [1] 0.2253597
```
The out of bag estimated sample error which is the estimated out of sample error based on training with repeated cross validation is 22.54% whereas the in sample error is 17.81% .
Additionally the out of sample error based on a seperate test (or validation) set is 25.69% .

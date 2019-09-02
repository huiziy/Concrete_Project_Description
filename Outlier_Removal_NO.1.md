OUTLIER REMOVAL
================
Huizi Yu
9/1/2019

GOAL for outlier detection:
---------------------------

STEPS:

-   Seperate the data into 80% training and 20% testing

-   Standarize the variables

-   Set up benchmark prior to removing outliers by calculating Rsquared using
    -   Linear Regression
    -   LASSO
    -   Random Forest
    -   NeuralNetwork
    -   XGboost
-   Try the different outlier detection methods in the basic outlier detection package in R
    -   density-based (den)
    -   knn (nn)
    -   ensemble (OutlierDetection)
    -   cooks distance

Loading Concrete Data into workplace
------------------------------------

``` r
setwd("~/Concrete_overdesign_PARIS")
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-16

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

``` r
library(abodOutlier)
```

    ## Loading required package: cluster

``` r
library(standardize)
library(OutliersO3)
library(OutlierDetection)
library(neuralnet)
library(HighDimOut)
library(caret)
library(tree)
library(gbm)
```

    ## Loaded gbm 2.1.5

``` r
library(xgboost)
library(knitr)
concrete <- read.csv("Clean_data.csv")
```

Randomly selecting one training set (80 %) and one testing set (20 %)
---------------------------------------------------------------------

``` r
set.seed(1234567+5*1000)
input <- concrete[,1:8]
input2 <- scale(input)
complete <- cbind(input2, concrete$overdesign)
concrete2 <- as.data.frame(complete) 
colnames(concrete2) <- c("coarse_agg_weight", "fine_agg_weight", "current_weight", "fly_ash_weight", "AEA_dose", "type_awra_dose", "weight_ratio", "target", "overdesign")
samp<-sample(1:nrow(concrete2),nrow(concrete2)*0.8,replace = F)
train <-concrete2[samp,]
test <- concrete2[-samp,]
mu <- mean(test$overdesign)
```

Removing non-important variables?
---------------------------------

``` r
tree_1 <- randomForest(y = concrete2$overdesign , x = concrete2[,1:8])
importance(tree_1, type = 2) 
```

    ##                   IncNodePurity
    ## coarse_agg_weight      47.92385
    ## fine_agg_weight        53.48026
    ## current_weight         49.65667
    ## fly_ash_weight         45.05715
    ## AEA_dose               53.77255
    ## type_awra_dose         71.22769
    ## weight_ratio           51.19938
    ## target                 40.40843

Random Forest picked coarse\_agg\_weight, fine\_agg\_weight, current\_weight, fly\_ash\_weight, AEA\_dose, type\_awra\_dose, W..C.P. all proved to be above 45, shows that we don't need to remove any variables.

Setting the benchmark before removing any outliers
==================================================

-   Methods
    -   Linear Regression
    -   LASSO
    -   Random Forest
    -   NeuralNetwork
    -   XGboost

### Linear Regression

``` r
y = train$overdesign ~ train$coarse_agg_weight + train$fine_agg_weight + train$current_weight + train$fly_ash_weight + train$AEA_dose + train$type_awra_dose + train$weight_ratio + train$target
linear <- lm(y)
Rsquared_linear <- summary(linear)$r.squared
if (Rsquared_linear == 0) {Rsquared_linear = 0}
Rsquared_linear
```

    ## [1] 0.2407882

### LASSO

``` r
cv.mod = glmnet (as.matrix(train[,1:8]),train$overdesign,alpha =1)
lasso.pred=predict (cv.mod,newx=as.matrix(test[,1:8]))
Rsquared_LASSO <- 1 - (sum((test$overdesign - lasso.pred)^2)/sum((test$overdesign - mu)^2))
if (Rsquared_LASSO <= 0) {Rsquared_LASSO = 0}
Rsquared_LASSO
```

    ## [1] 0

### Random Forest

``` r
tree <- randomForest(y = train$overdesign , x = train[,1:8], ntree = 500, importance = TRUE)
rf.pred <- predict(tree, newdata =as.matrix(test[,1:8]))
Rsquared_RF <- 1 - (sum((test$overdesign - rf.pred)^2)/sum((test$overdesign - mu)^2))
Rsquared_RF
```

    ## [1] 0.4578791

### NeuralNet

``` r
z = overdesign ~ coarse_agg_weight + fine_agg_weight + current_weight + fly_ash_weight + AEA_dose + type_awra_dose + weight_ratio + target
n <- neuralnet(z, data = train, hidden = 2, err.fct = 'sse', threshold = 0.1/0.5)
plot(n)
nn.pred <- compute(n, test[,1:8])
Rsquared_NN <- 1 - (sum((test$overdesign - nn.pred$net.result)^2)/sum((test$overdesign - mu)^2))
Rsquared_NN
```

    ## [1] 0.3500305

### XGBoost

``` r
train_values <- train[,1:8]
train_result <- train$overdesign
test_values <- test[,1:8]
test_result <- test$overdesign
dtrain = xgb.DMatrix(data =  as.matrix(train_values), label = train_result)
dtest = xgb.DMatrix(data =  as.matrix(test_values), label = test_result)
watchlist = list(train=dtrain, test=dtest)
xgb_train <- xgb.train(data = dtrain, 
                       max.depth = 8, 
                       eta = 0.3, 
                       nthread = 2, 
                       nround = 10000, 
                       watchlist = watchlist, 
                       objective = "reg:linear", 
                       early_stopping_rounds = 50,
                       print_every_n = 500)
```

    ## [1]  train-rmse:0.603671 test-rmse:0.601318 
    ## Multiple eval metrics are present. Will use test_rmse for early stopping.
    ## Will train until test_rmse hasn't improved in 50 rounds.
    ## 
    ## Stopping. Best iteration:
    ## [20] train-rmse:0.123723 test-rmse:0.165273

``` r
pred_val_xgboost <- predict(xgb_train, as.matrix(test[,1:8]))
Rsquared_XG <- 1 - (sum((test$overdesign - pred_val_xgboost)^2)/sum((test$overdesign - mu)^2))
Rsquared_XG
```

    ## [1] 0.4189204

The Benchmark table
-------------------

``` r
Benchmark <- as.data.frame(cbind(c("Linear Regression", "LASSO", "Random Forest", "Neural Network", "XGBoost"), c(Rsquared_linear,Rsquared_LASSO, Rsquared_RF, Rsquared_NN, Rsquared_XG)))
kable(Benchmark, format = "markdown", col.names = c("Method", "R-squared"))
```

| Method            | R-squared         |
|:------------------|:------------------|
| Linear Regression | 0.240788237889636 |
| LASSO             | 0                 |
| Random Forest     | 0.457879118012789 |
| Neural Network    | 0.35003045563753  |
| XGBoost           | 0.418920351934429 |

Basic Outlier Detection
=======================

### Density based outlier function

``` r
a <-dens(train)
removed_1 <- train[-a$`Location of Outlier`,]
newtrain <- removed_1
tree <- randomForest(y = newtrain$overdesign , x = newtrain[,1:8])
rf.pred <- predict(tree, newdata =as.matrix(test[,1:8]))
Rsquared_RF_2 <- 1 - (sum((test$overdesign - rf.pred)^2)/sum((test$overdesign - mu)^2))
Rsquared_RF_2
```

    ## [1] 0.4513968

### Knn outlier detection

``` r
b <- nn(train)
removed_2 <- train[-b$`Location of Outlier`,]
newtrain <- removed_2
## checking Rsquared with Random Forest
tree <- randomForest(y = newtrain$overdesign , x = newtrain[,1:8])
rf.pred <- predict(tree, newdata =as.matrix(test[,1:8]))
Rsquared_RF_3 <- 1 - (sum((test$overdesign - rf.pred)^2)/sum((test$overdesign - mu)^2))
Rsquared_RF_3
```

    ## [1] 0.4603604

### Ensemble outlier detection

``` r
c <- OutlierDetection(train)
removed_3 <- train[-c$`Location of Outlier`,]
newtrain <- removed_3
tree <- randomForest(y = newtrain$overdesign , x = newtrain[,1:8])
rf.pred <- predict(tree, newdata =as.matrix(test[,1:8]))
Rsquared_RF_4 <- 1 - (sum((test$overdesign - rf.pred)^2)/sum((test$overdesign - mu)^2))
Rsquared_RF_4
```

    ## [1] 0.4210016

### Cook's distance

``` r
cooksd <- cooks.distance(linear)
influential <- as.numeric(names(cooksd)[(cooksd > (4/nrow(train)))])
newtrain <- train[-influential,]

y = newtrain$overdesign ~ newtrain$coarse_agg_weight + newtrain$fine_agg_weight + newtrain$current_weight + newtrain$fly_ash_weight + newtrain$AEA_dose + newtrain$type_awra_dose + newtrain$weight_ratio + newtrain$target
linear2 <- lm(y)
Rsquared_linear_rm <- summary(linear2)$r.squared
if (Rsquared_linear_rm <= 0) {Rsquared_linear_rm = 0}
Rsquared_linear_rm
```

    ## [1] 0.2730367

``` r
tree <- randomForest(y = newtrain$overdesign , x = newtrain[,1:8])
rf.pred <- predict(tree, newdata =as.matrix(test[,1:8]))
Rsquared_RF_5 <- 1 - (sum((test$overdesign - rf.pred)^2)/sum((test$overdesign - mu)^2))
Rsquared_RF_5
```

    ## [1] 0.4105054

We increase Rsquared for Linear Regression by 3% but no significant improvement in Random Forest

The result table
----------------

``` r
result <- as.data.frame(cbind(c("dens", "nn", "OutlierDetection", "Cook distance"), c(Rsquared_RF_2, Rsquared_RF_3, Rsquared_RF_4, Rsquared_RF_5)))
kable(result, format = "markdown", col.names = c("Method", "R-squared"))
```

| Method           | R-squared         |
|:-----------------|:------------------|
| dens             | 0.451396782797922 |
| nn               | 0.460360367685005 |
| OutlierDetection | 0.42100160116157  |
| Cook distance    | 0.410505358884566 |

Conclusion
----------

So far, only KNN outlier detection method has marginally increased the Rsquared from 45% to 46%. For the next step, we attempt high dimension outlier removal method in the package "HighDimOut"

## Bart
##----------
library(dplyr)
library(caret)
library(randomForest)
library(rJava)
library(bartMachine)
library(MLmetrics)

## Data
data("bodyfat", package = "TH.data")

## Train/test
set.seed(3456)
trainIndex <- createDataPartition(bodyfat$DEXfat, p = .8,list=F)
train <- bodyfat[trainIndex,]
test  <- bodyfat[-trainIndex,]

## ML Models
## controls
fitControl <- trainControl(method = "repeatedcv",   
                           number = 5,     
                           repeats = 5)

### Random Forest
set.seed(1267)
rf_m<-train(DEXfat~.,
            data=train,
            method="ranger",
            trControl=fitControl)

### GBM
gbm_m<-train(DEXfat~.,
             data=train,
             method="gbm",
             trControl=fitControl)

### Bart
bartGrid <- expand.grid(num_trees = c(10, 15, 20, 100), k = 2, alpha = 0.95, beta = 2, nu = 3)
bart_m<-train(DEXfat~.,
             data=train,
             method="bartMachine",
             verbose = FALSE,
             tuneGrid = bartGrid, 
             trControl=fitControl)

### Cubist
grid <- expand.grid(committees = c(1, 10, 50, 100), neighbors = c(0, 1, 5, 9))
cubi_m<-train(DEXfat~.,
              data=train,
              method="cubist",
              tuneGrid = grid,
              trControl=fitControl)

#### Comparison
#################
## On train set
results <- resamples(list(rf=rf_m, 
                          gbm=gbm_m, 
                          bart=bart_m,
                          cubi=cubi_m))
bwplot(results,metric = "RMSE")
bwplot(results,metric = "MAE")
bwplot(results,metric = "Rsquared")

## On test set
pred_rf<-predict(rf_m,newdata = test)
pred_gbm<-predict(gbm_m,newdata = test)
pred_bart<-predict(bart_m,newdata = test)

RMSE(pred_rf,test$DEXfat)
RMSE(pred_gbm,test$DEXfat)
RMSE(pred_bart,test$DEXfat)

MAE(pred_rf,test$DEXfat)
MAE(pred_gbm,test$DEXfat)
MAE(pred_bart,test$DEXfat)

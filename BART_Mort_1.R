### Bayesian Additive Regression Trees
### Autor: Diego Lazo
### Fecha: 04/05/2022
#-------------------------------------------------
# Librerias
library(dplyr)
library(caret)
library(missForest)
library(data.table)
library(kableExtra)
library(bartMachine)
library(SoftBart)
library(MOTRbart)
library(gamair)
library(MLmetrics)
data("mpg")

# Functions
fun_mtrics<-function(x,train){
  rmse<-sqrt(sum((x-train)^2)/length(x))
  r2<-R2(x,train)
  mae<-MAE(x,train)
  df_r<-data.frame(RMSE=rmse,Rsquared=r2,MAE=mae) 
}

## Imputation
imp_res<-missForest(mpg)
imp_res$OOBerror
mpg_imp<-imp_res$ximp

# Feature Selection
## Recursive Feature Elimination
set.seed(123)
X=mpg_imp %>% select(-"hw.mpg")
y=mpg_imp %>% select("hw.mpg") %>% pull()
control <- rfeControl(functions=rfFuncs, method="repeatedcv",repeats = 3)
results <- rfe(x=X, 
               y=y,
               sizes=c(3,4,5,6,8), # number of features to iterate
               rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))
Xs=X[,names(X) %in% predictors(results)]
Xs$make<-as.numeric(Xs$make)
df_ml<-cbind(y,Xs)

# Train/test set
set.seed(3456)
trainIndex <- createDataPartition(df_ml$y, p = .8,list=F)
train <- df_ml[trainIndex,]
test  <- df_ml[-trainIndex,]
x_train<-train %>% select(-"y")
y_train<-train %>% select("y") %>% pull()
x_test<-test %>% select(-"y")

ctrl <- trainControl(method = "cv",number=5)

# XGboost
#--------
xgb_model <- train(x = x_train, 
                   y = y_train,
                   method = "xgbTree",
                   tuneLength =10,
                   trControl = ctrl)
pred_xgb <-predict(xgb_model, x_test)

#train rmse
tn_xgb<-sapply(xgb_model$resample[,1:3],mean)

#test rmse
tst_xgb<-postResample(pred_xgb,test$y)

# BART model
#-----------
bartGrid <- expand.grid(num_trees = c(10,20,50,100), k = 2,alpha=0.95,
                        beta = 2, nu = 3) 
bart_m<-train(x=x_train, 
              y=y_train, 
              method="bartMachine",
              tuneGrid = ctrl,
              trControl=fitControl,
              verbosity = 0)
bart_m
rmse_cv=min(bart_m$results$RMSE,na.rm = T)
y_pred_bart<-predict(bart_m,test)

#train rmse
tn_bart<-sapply(bart_m$resample[,1:3],mean)

#test rmse
tst_bart<-postResample(y_pred_bart,test$y)

# Model Tree BART - MOTbart
#--------------------------
set.seed(99)
motr_bart <- motr_bart(x=x_train, 
                       y=y_train, 
                       ntrees = bart_m$bestTune$num_trees,
                       nburn = 100,
                       npost = 100)
y_pred_motr <- predict_motr_bart(motr_bart, x_test,'mean')

#train rmse
df_train<-t(motr_bart$y_hat) %>% as.data.frame()
m_db<-rbindlist(lapply(df_train, fun_mtrics,y_train))  
tn_bart_mot<-sapply(m_db,mean,na.rm=TRUE)

#test rmse
tst_bart_mot<-data.frame(RMSE=RMSE(y_pred_motr,test$y),Rsquared=R2(y_pred_motr,test$y),
                         MAE=MAE(y_pred_motr,test$y))

## Resultados Finales
#--------------------
db_resfin_tn<-rbind(tn_xgb,tn_bart,tn_bart_mot) %>% round(3)
db_resfin_tst<-rbind(tst_xgb,tst_bart,tst_bart_mot) %>% round(3)
rownames(db_resfin_tn)<-c("XGboost","Bart","Bart_motr")
rownames(db_resfin_tst)<-c("XGboost","Bart","Bart_motr")

tf<-rbind(db_resfin_tn,db_resfin_tst)

kable(tf,align = 'c') %>%
  kable_styling(full_width = F) %>%
  pack_rows(index=c("Train CV metrics"=3,"Test metrics"=3)) 


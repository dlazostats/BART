## Bart
##----------
library(dplyr)
library(caret)
library(randomForest)
library(rJava)
library(mice)
library(readxl)
library(tidyverse)
library(psych)
library(bartMachine)
library(MLmetrics)
library(naniar)
library(outliers)
library(missForest)
library(Amelia)
library(VIM)

# Working directory
script_name <- 'BART_2.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Data
dbte0<-read_xlsx("tiempo_espera.xlsx") %>% as.data.frame()

# Preprocessing
## Imputation
dbte0$Planta<-factor(dbte0$Planta)
dbte0 %>% miss_var_summary() %>% as.data.frame()
dbte0 %>% spineMiss()
df_plot<-dbte0 %>% mutate(miss=ifelse(is.na(tdc_arribo),0,1))
df_plot_id<-dbte0 %>% rowid_to_column()
df_plot_id<-df_plot_id[is.na(df_plot_id$tdc_arribo),"rowid"]
plot(dbte0$tiempo_esp,dbte0$tdc_arribo)

imp_res<-missForest(dbte0)
imp_res$OOBerror
dbte2<-imp_res$ximp
dbte2_plo<-dbte2 %>% rowid_to_column()
dbte2_plo %>% 
  mutate(imputed=ifelse(dbte2_plo$rowid %in% df_plot_id,"imp","or")) %>% 
  ggplot(aes(x=tiempo_esp,y=tdc_arribo,color=imputed))+
  geom_point(alpha = 0.5)+
  scale_color_manual(values=c("#E69F00","grey90"))+
  theme_minimal()

## Anomaly detection
pairs.panels(dbte2[,c(1:5)])
dim(dbte2)
distances <- mahalanobis(x = dbte2[,c(1:5)] ,
                         center = colMeans(dbte2[,c(1:5)]),
                         cov = cov(dbte2[,c(1:5)]))
cutoff <- qchisq(p = 0.95 , df = ncol(dbte2)-1)
dbte3<-dbte2[distances < cutoff ,]
dim(dbte3)
(dim(dbte2)[1]-dim(dbte3)[1])/dim(dbte2)[1]
pairs.panels(dbte3[,c(1:5)])

## Train/test
set.seed(3456)
trainIndex <- createDataPartition(dbte3$tiempo_esp, p = .75,list=F)
train <- dbte3[trainIndex,]
test  <- dbte3[-trainIndex,]

## ML Model BART
#---------------
X<-train %>% select(-"tiempo_esp")
y<-train %>% select("tiempo_esp") %>% pull()
m_bart1<-bartMachine(X,y)
m_bart1
out_samp_s<-k_fold_cv(X, y, k_folds = 10)
out_samp_s$rmse
out_samp_s$PseudoRsq
rmse_bt<-rmse_by_num_trees(m_bart1, num_replicates = 20)

## CV
bart_machine_cv <- bartMachineCV(X, y)

## winning model
out_samp_s2<-k_fold_cv(X, y, k_folds = 10, k = 2, nu = 3, q = 0.9, num_trees = 200)

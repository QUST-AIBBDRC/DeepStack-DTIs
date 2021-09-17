library(DMwR)
require(methods)

setwd("E://A_SY")
data_train = read.csv("GPCR.csv",header = F)
data_train$V1=factor(data_train$V1)
train_data_SMOTEdata <- SMOTE(V1~.,data_train,perc.over =500,perc.under=120)
jishu<-table(train_data_SMOTEdata$V1)
write.csv(train_data_SMOTEdata,file='GPCR_smote.csv')

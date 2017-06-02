getwd()
rm(list=ls())
setwd("F:/Study/Financial data Science/project")


x<-read.csv("bank-full.csv",sep = ";")
x1<-read.csv("bank.csv",sep=";")

library(rattle)
library(rminer)

fit_1<-fit(y~.,data = x,model = "mlp",task="p",scale="inputs")
fit_1_1<-fit(y~.,data = x,model = "mlp",task="c",scale="inputs")
f_NN<-predict(fit_1,newdata=x1)
f_NN_1<-predict(fit_1_1,newdata=x1)
CF_NN<-data.frame(x1$y,nn=f_NN)
CF_NN_1<-data.frame(x1$y,nn=f_NN_1)
table(real=CF_NN_1[,1],NN=CF_NN_1[,2])
savemodel(fit_1,"NN")


fit_2<-fit(y~.,data = x,model = "svm",task="p",scale="inputs")
fit_2_2<-fit(y~.,data = x,model = "svm",task="c",scale="inputs")
f_SVM<-predict(fit_2,newdata=x1)
f_SVM_2<-predict(fit_2_2,newdata=x1)
CF_SVM<-data.frame(x1$y,svm=f_SVM)
CF_SVM_2<-data.frame(x1$y,svm=f_SVM_2)
table(real=CF_SVM_2[,1],SVM=CF_SVM_2[,2])
savemodel(fit_2,"SVM")


fit_3<-fit(y~.,data = x,model = "lr",task="p")
fit_3_3<-fit(y~.,data = x,model = "lr",task="c")
f_LR<-predict(fit_3,newdata=x1)
f_LR_3<-predict(fit_3_3,newdata=x1)
CF_LR<-data.frame(x1$y,lr=f_LR)
CF_LR_3<-data.frame(x1$y,lr=f_LR_3)
table(real=CF_LR_3[,1],LR=CF_LR_3[,2])
savemodel(fit_3,"LR")


fit_4<-fit(y~.,data = x,model = "dt",task="p")
fit_4_4<-fit(y~.,data = x,model = "dt",task="c")
f_DT<-predict(fit_4,newdata=x1)
f_DT_4<-predict(fit_4_4,newdata=x1)
CF_DT<-data.frame(x1$y,dt=f_DT)
CF_DT_4<-data.frame(x1$y,dt=f_DT_4)
savemodel(fit_4,"DT")
table(real=CF_DT_4[,1],DT=CF_DT_4[,2])



#------ROC---------------
ROC_NN<-mmetric(CF_NN[,1],CF_NN[,c(2,3),],"ROC")
AUC_NN<-ROC_NN$roc$auc
ROC_NN<-ROC_NN$roc$roc
plot(ROC_NN[,1],ROC_NN[,2],type="l",col="green",xlab="FPR",ylab="TPR")
ROC_SVM<-mmetric(CF_SVM[,1],CF_SVM[,c(2,3),],"ROC")
AUC_SVM<-ROC_SVM$roc$auc
ROC_SVM<-ROC_SVM$roc$roc
lines(ROC_SVM[,1],ROC_SVM[,2],type="l",col="red")
ROC_LR<-mmetric(CF_LR[,1],CF_LR[,c(2,3),],"ROC")
AUC_LR<-ROC_LR$roc$auc
ROC_LR<-ROC_LR$roc$roc
lines(ROC_LR[,1],ROC_LR[,2],type="l",col="blue")
ROC_DT<-mmetric(CF_DT[,1],CF_DT[,c(2,3),],"ROC")
AUC_DT<-ROC_DT$roc$auc
ROC_DT<-ROC_DT$roc$roc
lines(ROC_DT[,1],ROC_DT[,2],type="l",col="orange")
abline(0,1)
legend(0.6,0.4,c("NN","SVM","LR","DT"),fill =  c("green","red","blue","orange"))
#--------LIFT----------------
LIFT_NN<-mmetric(CF_NN[,1],CF_NN[,c(2,3),],"LIFT")
AUC_LIFT_NN<-LIFT_NN$lift$area
LIFT_NN<-LIFT_NN$lift$alift
plot(LIFT_NN[,1],LIFT_NN[,2],type="l",col="green",xlab="SAMPLE SIZE",ylab="RESPONSES")
LIFT_SVM<-mmetric(CF_SVM[,1],CF_SVM[,c(2,3),],"LIFT")
AUC_LIFT_SVM<-LIFT_SVM$lift$area
LIFT_SVM<-LIFT_SVM$lift$alift
lines(LIFT_SVM[,1],LIFT_SVM[,2],type="l",col="red")
LIFT_LR<-mmetric(CF_LR[,1],CF_LR[,c(2,3),],"LIFT")
AUC_LIFT_LR<-LIFT_LR$lift$area
LIFT_LR<-LIFT_LR$lift$alift
lines(LIFT_LR[,1],LIFT_LR[,2],type="l",col="blue")
LIFT_DT<-mmetric(CF_DT[,1],CF_DT[,c(2,3),],"LIFT")
AUC_LIFT_DT<-LIFT_DT$lift$area
LIFT_DT<-LIFT_DT$lift$alift
lines(LIFT_DT[,1],LIFT_DT[,2],type="l",col="orange")
abline(0,1)
legend(0.6,0.4,c("NN","SVM","LR","DT"),fill =  c("green","red","blue","orange"))

#----------IMPORTANCE ANALYSIS-------------------
IMP_NN<-Importance(fit_1,data = x,method = "SA")
IMP_NN_DSA<-Importance(fit_1,data = x,method = "DSA")
IMP_DT<-Importance(fit_4,data = x,method = "SA")
IMP_SVM<-Importance(fit_2,data = x,method = "SA")
IMP_LR<-Importance(fit_3,data = x,method = "SA")
vecplot(IMP_NN_DSA,sort = "decreasing",graph = "VEC",xlab = "Value range of attributes",ylab = "probability of outcome",main = "VEC curve")
sum(IMP_DT$value)
sum(IMP_SVM$value)
sum(IMP_LR$value)
IMP_NN_DSA$data
sum(IMP_NN$value)
IMP_NN$data
imp_table<-cbind.data.frame(features=colnames(x),relative_importance=IMP_NN$imp,SA=IMP_NN$value)
write.csv(imp_table,"relativeimportanceandSA.csv")


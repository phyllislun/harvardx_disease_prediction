####################################
# HarvardX PH125.9x Data Science: Capstone
# Choose Your Own Project Submission
# Objective: To create a machine learning model to predict diagnoses from symptom datast
# Author: Phyllis Lun
# Start date: 20220213
#
###################################
#
#1. an introduction/overview/executive summary section that describes the dataset and variables, and summarizes the goal of the project and key steps that were performed;
#2.	a methods/analysis section that explains the process and techniques used, including data cleaning, data exploration and visualization, insights gained, and your modeling approaches (you must use at least two different models or algorithms);
#3.	a results section that presents the modeling results and discusses the model performance; and
#4. a conclusion section that gives a brief summary of the report, its potential impact, its limitations, and future work.


#Executive summary
#The dataset for this Choose Your Own Project was taken from Kaggle (https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning), as shared by user KAUSHIL268.
#The dataset was designed for machine learning practice to predict 42 prognosis from 132 symptoms. The training set contains 4920 observations while the testing set contains 42. 
#The goal of this project is design a machine learning algorithm that is able to predict prognoses accurately from a variety of symptoms. 

#Methods 
#I. Data cleaning and exploration 

#First we are loading all the packages needed. 
library(tidyverse)
library(caret)
library(stringr)
library(psych)
library(knitr)
library(rmarkdown)
library(tinytex)


#library(stats)

#setwd("~/Dropbox/PL/HarvardX/Harvardx R/Rebourn R")

#Source of dataset: https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning


train_disease<- read.csv("/Users/phyllislun/Dropbox/PL/harvardX/Harvardx R/Rebourn R/disease prediction/Training.csv")
test_disease<- read.csv("/Users/phyllislun/Dropbox/PL/harvardX/Harvardx R/Rebourn R/disease prediction/Testing.csv")


#Exploring the dataset---
# There are 4920 observations in the training test and 42 in the test set.# Additionally there is a mismatch in terms of number of variables (column) in the test set (133) and in the training test (134.)
dim(train_disease)
dim(test_disease)
(names(train_disease)!=names(test_disease))
train_disease[,134]

head(train_disease) 
(test_disease$X) 
#There are 134 variables in the dataset, among which 133 are the symptoms, 1 is the prognosis. For some reason there is also an empty variable (X)
#The variable X is not available in the testing set
#Therefore, it would be best if we remove the empty variable and make sure that the test set and training set have matching colums. ----
summary(train_disease$X) #X is empty and has to be removed 
train_disease<-train_disease %>% select(-X)


#Symptoms: in a list -------
train_disease %>% select(-prognosis) %>% names()
symptoms<-train_disease %>% select(-prognosis) %>% names() %>% str_replace_all("[^[:alnum:]]", " ")
length(symptoms)
symptoms[symptoms=="fluid overload 1"]<-"fluid overload"


# Trying to rank symptoms by their number of endorsement
train_disease %>% select(-prognosis) %>% summarise_all(sum) %>% t() %>% as.data.frame() %>% rename("frequency"="V1")%>% arrange(desc(frequency)) %>% head(n=5)
train_disease %>% select(-prognosis) %>% summarise_all(sum) %>% t() %>% as.data.frame() %>% rename("frequency"="V1")%>% arrange(desc(frequency)) %>% tail(n=5)
#The most frequently reported symptoms are fatigue, vomit, high fever, loss of appetite and neasuea.
#The least frequently reported symptoms are pus filled  pimples, blackheads, scurring, foul smell of urine and fluid overload.

#Data visualisation: 
symptom.df<-train_disease%>% select(-prognosis) %>% summarise_all(sum) %>% t() %>% as.data.frame() %>% rename("frequency"="V1") 
symptom.df<-cbind(symptoms,symptom.df)
rownames(symptom.df)<-NULL

symptom.df %>% ggplot(aes(x = reorder(symptoms,frequency), y=frequency)) + 
  geom_bar(stat = "identity", size=1) + 
  labs(title="Endorsement of symptoms",x="Symptoms", y="Frequency", face="bold")+
  coord_flip () +
  theme_classic()+ 
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=12, face="bold"), 
        plot.title = element_text(size=14, face="bold"))
ggsave("~/Dropbox/PL/HarvardX/Harvardx R/Rebourn R/disease prediction/fig 1.png",dpi=300,height = 20, width = 8)


#Symptoms---- 
symptom_no<-train_disease %>% select(-prognosis) %>%rowSums(.) 
symptom.df.2<-as.data.frame(cbind(train_disease,symptom_no))
symptom.df.2 %>% group_by(prognosis) %>% summarise(n=mean(symptom_no)) %>% arrange((n)) %>% 
  ggplot(aes(x=reorder(prognosis,n),y=n))+
  geom_col() + 
  coord_flip () +
  theme_classic()+ 
  labs(title="Averaged number of symptoms by prognoses",x="Symptoms", y="Frequency", face="bold")+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=12, face="bold"), 
        plot.title = element_text(size=14, face="bold"))
  
#The prognoses-----------
#In the test set, there are 41 prognoses and 120 observations for each ratings. 
train_disease<-train_disease %>% mutate(prognosis=as.factor(prognosis))
test_disease<-test_disease %>% mutate(prognosis=as.factor(prognosis))
train_disease %>%  summarize(n_prognsis = n_distinct(prognosis)) 
train_disease %>% count(prognosis, sort=TRUE) 
test_disease %>% count(prognosis, sort=TRUE) 


#Here I verified that the training set and test set have the same prognoses
sum(levels(train_disease$prognosis)==levels(test_disease$prognosis))

#Machine learning model 1. Decision tree------------
library(rpart)

#2.1 Fine-tuning the decision tress----
set.seed(3, sample.kind = "Rounding")
fit_rpart <- train(prognosis ~ .,
                    method = "rpart",
                    tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 200)),
                    data = train_disease)

#plotting the cp parameter tuning 
plot(fit_rpart)
fit_rpart$bestTune #The best cp value selected from boostrapping is 0.001005025.

fit_rpart$finalModel$variable.importance
#In the final model, it is found that mild fever, headache, sweating,yellowing of eyes, and muscle weakness were the most important features. 

#plotting the decision tree [not displayed in Rmd]
#png("~/Dropbox/PL/HarvardX/Harvardx R/Rebourn R/disease prediction/decision tree.png", 
#    width = 2400, height = 1200)

#plot(fit_rpart$finalModel, margin = 0.1, frame = FALSE, 
#     main="Decision tree for prognosis prediction", 
#     cex.main=1.2)
#text(fit_rpart$finalModel, cex = 0.6)
#dev.off()

#accuracy in the test set
y_hat <- predict(fit_rpart, test_disease)
dt_accuracy<-confusionMatrix(y_hat, as.factor(test_disease$prognosis))$overall["Accuracy"] 

#The classification tree reaches an accuracy of 0.8. 
disease_results <- tibble(method = "Decision tree", accuracy = dt_accuracy)
disease_results %>% knitr::kable()

#2. Random Forest-----
library(randomForest)
train_disease<-train_disease %>% mutate(prognosis=as.factor(prognosis))
set.seed(3, sample.kind = "Rounding")
rt_fit <- randomForest(prognosis ~ ., data = train_disease)

#Feature importance
rt_fit$ntree #500
rt_fit$mtry #11
rt_fit$predicted

varImpPlot(rt_fit, main="Feature importance of the prognosis prediction model") 
#The first most important features are muscle pain, altered sensorium, family history, mild fecer, and chest pain. 

#accuracy in the test set
y_hat <- predict(rt_fit, test_disease)
rf_accuracy<-confusionMatrix(y_hat, as.factor(test_disease$prognosis))$overall["Accuracy"] #0.9761905

disease_results <-bind_rows(disease_results, data.frame(method="Random forest model 1", accuracy=rf_accuracy))
disease_results %>% knitr::kable()


#3. using the train commend -- to tune the random forest model

#v1 In this random forest model, we used cross validation (5 times and sampling 20% of the training set observations) to determine the best models from varying number of 
#unique samples needed to split nodes and number of predictors needed. 
library(Rborist)
train_disease_x<-train_disease%>% select(-prognosis)
train_disease_y<-train_disease%>% select(prognosis) %>% pull(.)%>% as.factor()

set.seed(3, sample.kind = "Rounding")

control <- trainControl(method="cv", number = 5, p = 0.2) #adding cross validation
grid <- expand.grid(minNode=c(2,3,4,5),  predFixed =seq(2:10))
train_rf <-  train(train_disease_x, 
                   train_disease_y, 
                   method = "Rborist", 
                   nTree = 500,
                   trControl = control,
                   tuneGrid = grid)

#Displaying the random forest results
ggplot(train_rf)+ 
  labs(title = "Accurancy of random forest model with different number\nof predictors, stratified by minimal number of nobes")+
  theme_classic()+ 
  scale_x_continuous("Number of randomly selected predictors") +
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=12, face="bold"), 
        plot.title = element_text(size=14, face="bold"))
#The most optimal model, yielding the highest accuracy, uses two predictors and a minimum of 2 observations.
varImp(train_rf)

#The most important variables are headache, vomiting, stiff neck, yellowing of eyes, and abdominal pain.
y_hat <- predict(train_rf, test_disease)
rf_tuned_accuracy<-confusionMatrix(y_hat, as.factor(test_disease$prognosis))$overall["Accuracy"] #0.9761905
#Unfortunately, the fine-tuned random model did not improve on the accuracy of the first random forest model. 
disease_results <-bind_rows(disease_results, data.frame(method="Random forest model 2", accuracy=rf_tuned_accuracy))
disease_results %>% knitr::kable()


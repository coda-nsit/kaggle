setwd('/home/bloodynacho/Documents/Kaggle/Titanic/')
source('train.R')
source('test.R')

train <- read.csv('train.csv')
train <- cleanTrain(train)
theta <- computeTheta(train)

test <- read.csv('test.csv')
x <- test
test <- cleanTest(test)
answer <- testFeatures(test, theta)
answer <- as.numeric(answer >= 0)
answer <- cbind(x$PassengerId, answer)
head(answer)
answer <- data.frame(answer)
write.csv(answer, file="answer.csv", eol='\r', row.names=FALSE)

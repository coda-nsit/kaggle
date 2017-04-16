library(zoo)
cleanTest <- function(test){
  test$Sex <- as.numeric(test$Sex == "male")
  drops <- c("PassengerId", "Name", "Ticket", "Cabin")
  test <- test[ , !(names(test) %in% drops)]
  test$Embarked <- sapply(test$Embarked, emb)
  # test <- test[complete.cases(test), ]
  test <- na.aggregate(test)
  col_means <- apply(test, 2, mean, na.rm=TRUE)
  col_sd <- apply(test, 2, sd, na.rm=TRUE)
  
  
  
  x <- test - col_means
  test <- t(apply(x, 1, function(x) x / col_sd))
  test <- cbind(test, rep(1, nrow(test)))
  colnames(test)[8] <- "x0"
  test
}


testFeatures <- function(test, theta){
  # print(head(theta))
  # print(head(test))
  test %*% theta
}

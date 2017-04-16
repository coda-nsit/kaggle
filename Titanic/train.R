library(ggplot2)
emb <- function(x){
  y <- c()
  for(i in x){
    if(i=='C'){
      y <- c(y, 0) 
    }
    else if(i=='Q'){
      y <- c(y, 1)
    }
    else if(i=='S'){
      y <- c(y, 2)
    }
    else{
      y <- c(y, NA)
    }
  }
  y
}


sigmoid <- function(x){
  1 / (1 + exp(-x))
}


computeCost <- function(theta, x, y){
  h <- sigmoid(x %*% theta)
  -1 / m * (t(y) %*% log(h) + t(1-y) %*% log(1-h))  
}


cleanTrain <- function(train){
  train$Sex <- as.numeric(train$Sex == "male")
  drops <- c("PassengerId", "Name", "Ticket", "Cabin")
  train <- train[ , !(names(train) %in% drops)]
  train$Embarked <- sapply(train$Embarked, emb)
  train <- train[complete.cases(train), ]
  y <- train[, 'Survived']
  drops <- c("Survived")
  train <- train[ , !(names(train) %in% drops)]
  col_means <- apply(train, 2, mean)
  col_sd <- apply(train, 2, sd)
  x <- train - col_means
  train <- t(apply(x, 1, function(x) x / col_sd))
  train <- cbind(train, rep(1, nrow(train)))
  colnames(train)[8] <- "x0"
  train
}


computeTheta <- function(train){
  m <- nrow(train)
  n <- ncol(train)
  theta <- matrix(0, nrow=n, ncol=1)
  
  num_iterations <- 10000
  alpha <- 0.001
  cost <- c()
  gradients <- matrix(nrow=0, ncol=n)
  for(i in 1:num_iterations){
    h <- sigmoid(train %*% theta)
    gradient <- (t(train) %*% (h - y)) / m
    gradients <- rbind(gradients, t(gradient))
    theta = theta - alpha * gradient
    cost <- c(cost, computeCost(theta, train, y))
  }
  iter <- c(1:length(cost))
  plot(iter, cost, xlab="iterations", ylab="cost")
  theta
}
  
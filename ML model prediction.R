## ---------------------------------------------------------------
## Breast Cancer Classification & Clustering
## Compares Logistic Regression, KNN, Random Forest, Decision Tree,
## and SVM on the Wisconsin Breast Cancer dataset, plus K-means /
## hierarchical clustering and PCA for exploration.
## ---------------------------------------------------------------

library(caTools)
library(caret)
library(class)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)

## ---------------------------------------------------------------
## 1. Load & clean data
## ---------------------------------------------------------------
data <- read.csv("BreastCancer.csv")
data$Id <- NULL
data <- na.omit(data)
data$Class <- as.factor(data$Class)

## Identify predictor columns explicitly (safer than hardcoded indices)
predictor_cols <- setdiff(names(data), "Class")

## ---------------------------------------------------------------
## 2. Train / test split (stratified on the response)
## ---------------------------------------------------------------
set.seed(123)
split <- sample.split(data$Class, SplitRatio = 0.90)   # FIX: split on Class, not the whole data frame
train_data <- subset(data, split == TRUE)
test_data  <- subset(data, split == FALSE)

## ---------------------------------------------------------------
## 3. Logistic Regression
## ---------------------------------------------------------------
logit_data <- glm(Class ~ ., data = train_data, family = "binomial")  # FIX: train on train_data, not full data

## Visualize fitted probability curve vs. Cell.size
plot(data$Cell.size, data$Class,
     xlab = "Cell size", ylab = "Class",
     xlim = c(0, 30), ylim = c(0, 1),
     col = "red", pch = 1, cex = 2)

av <- seq(0, 30, 1)   # FIX: match plot's xlim so the curve isn't truncated

a <- coefficients(logit_data)["(Intercept)"]
b <- coefficients(logit_data)["Cell.size"]

pv <- 1 / (1 + exp(-1 * (a + b * av)))
lines(av, pv, col = "blue", lwd = 2)

threshold <- -(a / b)
abline(v = threshold, lty = 2, col = "grey", lwd = 2)

prediction <- predict(logit_data, test_data, type = "response")
c1 <- ifelse(prediction > 0.5, 1, 0)
c1 <- factor(c1, levels = levels(test_data$Class))  # FIX: align factor levels with the true labels

confusionmatrix_lreg <- confusionMatrix(c1, test_data$Class)
confusionmatrix_lreg

## ---------------------------------------------------------------
## 4. K-Nearest Neighbors
## ---------------------------------------------------------------
set.seed(123)
train_scale <- scale(train_data[, predictor_cols])
test_scale  <- scale(test_data[, predictor_cols])

knnmodel <- knn(train = train_scale, test = test_scale,     # FIX: use scaled, predictor-only matrices
                 cl = train_data$Class, k = 4)

confusionmatrix_knn <- confusionMatrix(knnmodel, test_data$Class)  # FIX: prediction first, reference second
confusionmatrix_knn

## ---------------------------------------------------------------
## 5. K-means clustering (exploratory)
## ---------------------------------------------------------------
km_data <- data[, c("Cell.shape", "Cell.size")]
set.seed(123)
km2 <- kmeans(km_data, 2, nstart = 50)
km3 <- kmeans(km_data, 3, nstart = 50)
km4 <- kmeans(km_data, 4, nstart = 50)
km5 <- kmeans(km_data, 5, nstart = 50)
km6 <- kmeans(km_data, 6, nstart = 50)
km7 <- kmeans(km_data, 7, nstart = 50)
km8 <- kmeans(km_data, 8, nstart = 50)
km9 <- kmeans(km_data, 9, nstart = 50)

plot(km_data, col = km9$cluster, pch = 20, cex = 3)

w <- c(km2$tot.withinss, km3$tot.withinss, km4$tot.withinss, km5$tot.withinss,
       km6$tot.withinss, km7$tot.withinss, km8$tot.withinss, km9$tot.withinss)
barplot(w, names.arg = seq(2, 9), xlab = "k", ylab = "Within-cluster SS")

c1_km <- which(km9$cluster == 1)
cluster1 <- data[c1_km, ]

## ---------------------------------------------------------------
## 6. Hierarchical clustering
## ---------------------------------------------------------------
set.seed(240)
g.lab <- data[, predictor_cols]
g.sc <- scale(g.lab)
g.dist <- dist(g.sc, method = "euclidean")

hc  <- hclust(g.dist, method = "average")
hc1 <- hclust(g.dist, method = "complete")

plot(hc, labels = data$Class, main = "Average linkage")
abline(h = 3, col = "red")
fit <- cutree(hc, k = 3)
rect.hclust(hc, k = 3, border = "green")

plot(hc1, labels = data$Class, main = "Complete linkage")  # FIX: hc1 was computed but never used

## ---------------------------------------------------------------
## 7. PCA
## ---------------------------------------------------------------
d <- data[, predictor_cols]
d.pca <- prcomp(d, center = TRUE, scale. = TRUE)
summary(d.pca)

pv_pca <- d.pca$sdev^2 / sum(d.pca$sdev^2)
plot(pv_pca, xlab = "Principal Component", ylab = "Proportion of Variance Explained",
     pch = 20, cex = 2, col = 4)

cpv <- cumsum(pv_pca)
plot(cpv, xlab = "Principal Component", ylab = "Cumulative Proportion of Variance",
     pch = 20, cex = 2, col = 4)

d.load  <- d.pca$rotation
d.score <- d.pca$x
summary(d.score)

## ---------------------------------------------------------------
## 8. Random Forest
## ---------------------------------------------------------------
set.seed(123)
rf <- randomForest(Class ~ ., data = train_data, ntree = 500, importance = TRUE)
prediction_rf <- predict(rf, test_data)
confusionmatrix_rf <- confusionMatrix(prediction_rf, test_data$Class)
confusionmatrix_rf
varImpPlot(rf)

## ---------------------------------------------------------------
## 9. Decision Tree
## ---------------------------------------------------------------
set.seed(123)
dt <- rpart(Class ~ ., data = train_data, method = "class")
prediction_dt <- predict(dt, test_data, type = "class")
confusionmatrix_dt <- confusionMatrix(prediction_dt, test_data$Class)
confusionmatrix_dt

printcp(dt)
rpart.plot(dt, box.palette = "auto", nn = TRUE)

## ---------------------------------------------------------------
## 10. SVM
## ---------------------------------------------------------------
set.seed(123)
svm_model <- svm(Class ~ ., data = train_data,
                  type = "C-classification", kernel = "radial", gamma = 0.1)
prediction_svm <- predict(svm_model, test_data)
confusionmatrix_svm <- confusionMatrix(prediction_svm, test_data$Class)
confusionmatrix_svm

tuned_svm <- tune.svm(Class ~ ., data = train_data,
                       type = "C-classification", kernel = "radial", gamma = 0.1)
tuned_svm$best.parameters
tuned_svm$best.performance

best_svm <- tuned_svm$best.model
prediction_svm_best <- predict(best_svm, test_data)
confusionmatrix_best_svm <- confusionMatrix(prediction_svm_best, test_data$Class)
confusionmatrix_best_svm
plot(tuned_svm)

## ---------------------------------------------------------------
## 11. Model comparison summary
## ---------------------------------------------------------------
model_accuracy <- c(
  Logistic_Regression = confusionmatrix_lreg$overall["Accuracy"],
  KNN                  = confusionmatrix_knn$overall["Accuracy"],
  Random_Forest        = confusionmatrix_rf$overall["Accuracy"],
  Decision_Tree        = confusionmatrix_dt$overall["Accuracy"],
  SVM                  = confusionmatrix_svm$overall["Accuracy"],
  SVM_Tuned            = confusionmatrix_best_svm$overall["Accuracy"]
)
print(model_accuracy)

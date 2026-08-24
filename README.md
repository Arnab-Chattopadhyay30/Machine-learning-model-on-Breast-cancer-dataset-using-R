# Breast Cancer Classification & Clustering (R)

This project applies several classification and unsupervised learning techniques to the Wisconsin Breast Cancer dataset in order to predict whether a tumor is benign or malignant, and to explore structure in the underlying feature space.

## Dataset

The data used here is the Wisconsin Breast Cancer dataset, commonly distributed through the [`mlbench`](https://cran.r-project.org/package=mlbench) package as `BreastCancer`, or available for download from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). The script expects a file named `BreastCancer.csv` in the same directory it is run from; if your file lives elsewhere, update the `read.csv()` path accordingly. On load, the `Id` column is dropped, rows containing missing values are removed with `na.omit()`, and the target variable, `Class`, is cast to a factor for modeling. If you don't already have the CSV on hand, you can generate one directly in R with `library(mlbench); data(BreastCancer); write.csv(BreastCancer, "BreastCancer.csv", row.names = FALSE)`.

## What the script does

`breast_cancer_classification.R` runs end to end, starting with data loading and cleaning, followed by a 90/10 train/test split that is stratified on `Class` using `caTools::sample.split`. From there it fits a logistic regression model on the training data, plots the fitted probability curve against `Cell.size`, and evaluates it on the held-out test set. Next comes a k-nearest neighbors model (k = 4), with features scaled before the distance calculation so no single variable dominates. The script then moves into unsupervised exploration: k-means clustering is run for k = 2 through 9 on `Cell.shape` and `Cell.size`, with a within-cluster sum-of-squares plot to help visually identify an elbow, followed by hierarchical clustering using both average and complete linkage on scaled features, and a principal component analysis that reports variance explained and cumulative variance explained across components. Supervised modeling resumes with a 500-tree random forest and a variable importance plot, a decision tree built with `rpart` and visualized with `rpart.plot`, and finally a support vector machine with a radial kernel, both at a fixed gamma and after tuning via `tune.svm`. The script closes by printing a summary table comparing test accuracy across all of the classifiers.

## Requirements

The script depends on `caTools`, `caret`, `class`, `randomForest`, `rpart`, `rpart.plot`, and `e1071`, which can all be installed with `install.packages(c("caTools", "caret", "class", "randomForest", "rpart", "rpart.plot", "e1071"))`. It has been tested with R 4.x; note that `caret` will pull in a number of additional dependencies, such as `lattice` and `ggplot2`, on first install.

## Running

Run the script from the command line with `Rscript breast_cancer_classification.R`, or open it in RStudio and step through it interactively. Several `plot()` calls are included throughout and are meant to be viewed one at a time; if you'd prefer saved output instead of interactive plots, wrap the relevant sections in a PDF or PNG device call.

## Notes on fixes from the original draft

A few issues were corrected from an earlier version of this script. The logistic regression model is now trained only on `train_data`, whereas it was previously fit on the full dataset, which let test rows leak into training. The `sample.split()` call now stratifies on `data$Class` rather than the whole data frame, so the split behaves as intended. The KNN model now runs on scaled, predictor-only matrices; previously the scaled versions of the data were computed but never actually used, and the `Class` column was inadvertently included in the distance calculation. The `confusionMatrix()` calls are now consistent in argument order (`prediction` first, `reference` second) across every model, so that sensitivity and specificity are directly comparable. The complete-linkage hierarchical clustering result, which was computed but never plotted, is now actually visualized. Finally, the x-range used to draw the logistic regression curve now matches the plot's `xlim`, so the curve no longer appears truncated.

## Possible next steps

Future improvements could include cross-validation, for example using `caret::train` with a `trainControl` object, in place of a single train/test split to get more robust accuracy estimates; adding ROC and AUC curves for each model alongside the existing confusion matrices; and using the PCA loadings and random forest variable importance results to guide feature selection.

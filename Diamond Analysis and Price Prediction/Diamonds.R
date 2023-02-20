setwd("/Users/riccardopandolfi/Desktop/POLIMI/STATISTICS/Project")
library(readr)
diamonds <- read_csv("diamonds.csv")

#data clean
diamonds[diamonds==0] <- NA
diamonds<-diamonds[complete.cases(diamonds),]


# 1) Descriptive analysis of the dataset
summary(diamonds)

#Now, the max price of the diamonds seemed pretty low so we decided to do a scatterplot to see
#how this was possible
#To see which variables to use in the scatterplot we decided to take two variables that
#were significant for the analysis and highly correlated
library(GGally)
ggcorr(diamonds, label = T)

library(ggplot2)

data("diamonds")

ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 0.5) + labs(x = "Carat", y = "Price")

#Now let's keep only relevant obs
diamonds <- diamonds[which(diamonds$carat <= 3),]

summary(diamonds)
hist(diamonds$carat, xlab = "Carat", main = "Carat")
hist(diamonds$depth, xlab = "Depth", main = "Depth Variable", col="purple")
hist(diamonds$price, xlab = "Price", main = "Price Variable", col="gold")
# of course it is a valid analysis only for the quantitative variables

# Now, we have to consider also the other qualitative variables 
attach(diamonds)
library(modeest)
m<-c(cut)
freq_table <- table(m)
mode <- names(which.max(freq_table))
mode
barplot(freq_table, main="Frequency of Cut Types", xlab="Class", ylab="Frequency", col="pink")

col<-c(color)
freq_table2 <- table(col)
mode2 <- names(which.max(freq_table2))
mode2
barplot(freq_table2, main="Frequency of Color Type", xlab="Class", ylab="Frequency", col="red")

clar<-c(clarity)
freq_table3 <- table(clar)
mode3 <- names(which.max(freq_table3))
mode3
barplot(freq_table3, main="Frequency of Clarity Scores", xlab="Class", ylab="Frequency", col="purple")


#there seems to be a "cut" dataset: the max price is set in the dataset and we have no more 
#diamomds up that treshold. 
#Not good

# 2) Identikit of the highest quality diamonds. The highest quality diamonds are those
# which have a clarity grade equal to IF (Internally Flawless). 
# First thing to do is to create a new dataframe that we will use for this analysis
clarity_IF <- subset(diamonds, clarity == "IF")
# since they are really rare diamonds here we have only 1790 observations

# Now, let's see the price range for these diamonds

min(clarity_IF$price)
max(clarity_IF$price)

mean(clarity_IF$price)
sd(clarity_IF$price)

IQR(clarity_IF$price)

# Same with carat

min(clarity_IF$carat)
max(clarity_IF$carat)

mean(clarity_IF$carat)
sd(clarity_IF$carat)

IQR(clarity_IF$carat)

#Now let's see usually which color it has
IF_col<-c(clarity_IF$color)
IF_freq_table <- table(IF_col)
IF_mode <- names(which.max(IF_freq_table))
IF_mode
barplot(IF_freq_table, main="Frequency of Color Classes IF case", xlab="Class", ylab="Frequency", col="red")
par(mfrow=c(1,2))
barplot(IF_freq_table, main="Frequency of Color Classes in the IF case", xlab="Class", ylab="Frequency", col="red")
barplot(freq_table2, main="Frequency of Color Classes in general", xlab="Class", ylab="Frequency", col="pink")

#Let's do the same with the cut
IF_cut<-c(clarity_IF$cut)
IF_freq_table2 <- table(IF_cut)
IF_mode2 <- names(which.max(IF_freq_table2))
IF_mode2
barplot(IF_freq_table2, main="Frequency of Cut Classes", xlab="Class", ylab="Frequency", col="green")
par(mfrow=c(1,2))
barplot(IF_freq_table2, main="Frequency of Cut Classes in the IF case", xlab="Class", ylab="Frequency", col="green")
barplot(freq_table, main="Frequency of Cut Classes in general", xlab="Class", ylab="Frequency", col="blue")

# Now, let's perform a correlation analysis to see how and which are the variables correlated
str(diamonds)
diam_quant <- diamonds [,c(1,5,6,7,8,9,10)]
library(corrplot)
library(RColorBrewer)
corr_quant <-cor(diam_quant)
corr_quant
corrplot(corr_quant, type="upper", order="hclust", col=brewer.pal(n=8, name="RdYlBu"))


#PREDICTION MODEL FOR PRICES
library(readr)
diamonds <- read_csv("diamonds.csv")

#CUT
diamonds$DummyGood  <- ifelse(diamonds$cut == 'Good', 1, 0)
diamonds$DummyIdeal <- ifelse(diamonds$cut == 'Ideal', 1, 0)
diamonds$DummyPremium <- ifelse(diamonds$cut == 'Premium', 1, 0)
diamonds$DummyVeryGood <- ifelse(diamonds$cut == 'Very Good', 1, 0)

#COLOR
diamonds$DummyE <- ifelse(diamonds$color == 'E', 1, 0)
diamonds$DummyF <- ifelse(diamonds$color == 'F', 1, 0)
diamonds$DummyG <- ifelse(diamonds$color == 'G', 1, 0)
diamonds$DummyH <- ifelse(diamonds$color == 'H', 1, 0)
diamonds$DummyI <- ifelse(diamonds$color == 'I', 1, 0)
diamonds$DummyJ <- ifelse(diamonds$color == 'J', 1, 0)

#CLARITY
diamonds$DummyVS2 <- ifelse(diamonds$clarity == 'VS2', 1, 0)
diamonds$DummySI2 <- ifelse(diamonds$clarity == 'SI2', 1, 0)
diamonds$DummyVS1 <- ifelse(diamonds$clarity == 'VS1', 1, 0)
diamonds$DummyVVS2 <- ifelse(diamonds$clarity == 'VVS2', 1, 0)
diamonds$DummyVVS1 <- ifelse(diamonds$clarity == 'VVS1', 1, 0)
diamonds$DummyIF <- ifelse(diamonds$clarity == 'IF', 1, 0)
diamonds$DummyI1 <- ifelse(diamonds$clarity == 'I1', 1, 0)

#Create the dataset, response in the last column, load useful libraries
diamonds_no_pca <- cbind(diamonds[, c(2,6,7,9:28)], diamonds[8]) #this is the dataset used for the model, NO PCA
library(glmnet)
library(caret)

# extract the predictors and response
X <- diamonds_no_pca[1:23]
y <- diamonds_no_pca$price

# convert predictors to a matrix
X_matrix <- as.matrix(X)

# specify the number of folds for cross-validation
k <- 10

# set the seed for reproducibility
set.seed(123)

# split the data into training and testing sets
train_indices <- createDataPartition(y, times = 1, p = 0.8, list = FALSE)
X_train <- X_matrix[train_indices, ]
y_train <- y[train_indices]
X_test <- X_matrix[-train_indices, ]
y_test <- y[-train_indices]

# fit a lasso regression model using cross-validation to tune lambda
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = k, 
                         lambda = seq(0.001, 1, length = 100))

# choose the best value of lambda
best_lambda <- lasso_model$lambda.min

# refit the model using the best lambda value
final_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)

# extract the coefficients
coef_final_model <- coef(final_model)

# view the coefficients in a data frame
coef_df <- data.frame(variable = rownames(coef_final_model)[-1],
                      coefficient = coef_final_model[-1])

# order by coefficient magnitude
coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE),]

# view the coefficients
print(coef_df)

# use k-fold cross-validation to estimate the test error
cv_results <- cv.glmnet(X_matrix, y, alpha = 1, nfolds = k, 
                        lambda = seq(0.001, 1, length = 100))
cv_rmse_test <- sqrt(cv_results$cvm[cv_results$lambda == best_lambda])

# calculate the root mean squared error on the training set
y_train_pred <- predict(final_model, newx = X_train)
rmse_train <- sqrt(mean((y_train - y_train_pred)^2))

# calculate the root mean squared error on the test set
y_test_pred <- predict(final_model, newx = X_test)
rmse_test <- sqrt(mean((y_test - y_test_pred)^2))

# view the root mean squared errors
print(paste("RMSE on training set:", round(rmse_train, 2)))
print(paste("RMSE on test set:", round(rmse_test, 2)))
print(paste("RMSE estimated with cross-validation:", round(cv_rmse_test, 2)))

# view the plot of the model
library(ggplot2)

# create a data frame with actual and predicted values for the test set
results_test <- data.frame(actual = y_test, predicted = as.vector(y_test_pred))

# create a scatter plot of actual vs predicted values for the test set
ggplot(results_test, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 1, color = "red") +
  labs(x = "Actual Price", y = "Predicted Price", title = "Actual vs Predicted Prices - Test Set") +
  theme(panel.background = element_rect(fill = "white"), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"))

# create a data frame with actual and predicted values for the training set
results_train <- data.frame(actual = y_train, predicted = as.vector(y_train_pred))

# create a scatter plot of actual vs predicted values for the training set
ggplot(results_train, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 1, color = "red") +
  labs(x = "Actual Price", y = "Predicted Price", title = "Actual vs Predicted Prices - Training Set") +
  theme(panel.background = element_rect(fill = "white"), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"))

# calculate adjusted R-squared on the test set
library(caret)
rsq_test <- R2(y_test_pred, y_test)
n_test <- length(y_test)
p_test <- ncol(X_test) - 1 
adj_rsq_test <- 1 - (1 - rsq_test) * (n_test - 1) / (n_test - p_test - 1)
adj_rsq_test

# calculate R-squared on the training set
rsq_train <- R2(y_train_pred, y_train)
n_train <- length(y_train)
p_train <- ncol(X_train) - 1 
adj_rsq_train <- 1 - (1 - rsq_train) * (n_train - 1) / (n_train - p_train - 1)
adj_rsq_train
------------------------------------------------------------------------

  








        


        
                       








  





















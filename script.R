#############################
# Predicting video game sales
#############################

###############
# Set Up
###############

# install any required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")


# load libraries
library(tidyverse)
library(gridExtra)
library(Hmisc)
library(caret)
library(kableExtra)
library(glmnet)
library(randomForest)


# download raw data file from git repo
URL <- tempfile()
download.file("https://github.com/nari9/Harvard_VideoGame/raw/master/videogames_data.csv",URL)
data <- read.csv(file=URL)



##################
# Data Cleaning
##################

# view raw data file
data %>% as_tibble()


# filter out rows post 2016
# NAs introduced will be removed, warning can be ignored
data <- data %>% filter((as.numeric(as.character(data$Year_of_Release)))<=2016)


# Update User_Score column to numeric
# NAs introduced will be removed, warning can be ignored
data$User_Score <- as.numeric(as.character(data$User_Score))


# overview of columns with NA
colSums(is.na(data))


# remove rows that have any NA values
data <- na.omit(data)


# overview of columns with blanks
colSums(data == '')


# remove rows with blank data
game_sales <- data %>% filter(Developer != '' & Rating != '')

# remove data object
rm(data)


####################################
# Data Exploration and Visualization
####################################


# plot total sales per platform
platform_sales <- game_sales %>% 
  group_by(Platform) %>% 
  summarise(NA_Sales = sum(NA_Sales)) %>% ggplot(aes(x = Platform, y = NA_Sales)) + 
  geom_bar(stat="identity", fill="steelblue") +
  ylab("North America Sales (million units)") +
  theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"),
        axis.line = element_line(colour = "black")) +
  ggtitle("Video Game Sales per Platform")
platform_sales


# times series plot - Group sales data by release year
year_data <- game_sales %>% group_by(Year_of_Release) %>% 
  summarise(NA_Sales = sum(NA_Sales))

# times series plot - Update release year data to be numeric
year_data$Year_of_Release <- as.numeric(as.character(year_data$Year_of_Release))

# time series plot of sales
time_plot <- year_data %>%
  ggplot(aes(Year_of_Release, NA_Sales)) +
  geom_line() +
  ylab("North America Sales (million units)") + 
  theme_bw()
time_plot


# plot of genre vs sales
genre_plot <- game_sales %>% 
  group_by(Genre) %>% 
  summarise(NA_Sales = sum(NA_Sales)) %>% 
  ggplot(aes(x = Genre, y = NA_Sales)) + 
  geom_bar(stat="identity", fill="steelblue") +
  ylab("North America Sales (million units)") +
  theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"),
        axis.line = element_line(colour = "black")) +
  ggtitle("Video Games Sales per Genre") +
  theme(axis.text.x = element_text(angle = 90))
genre_plot


# plot of critic score vs avg sales
critic_plot <- game_sales %>% 
  group_by(Critic_Score) %>% 
  summarise(avg = mean(NA_Sales)) %>% 
  ggplot(aes(Critic_Score, avg)) + 
  geom_point() + 
  ylab("Average Sales") +
  theme_bw()


# plot of user score vs avg sales
user_plot <- game_sales %>% 
  group_by(User_Score) %>% 
  summarise(avg = mean(NA_Sales)) %>% 
  ggplot(aes(User_Score, avg)) + 
  geom_point() + 
  ylab("Average Sales") +
  theme_bw()

# arrange plots of critic and user score vs sales
grid.arrange(critic_plot, user_plot, ncol=2)


# top 10 best games not taking platform into account
best_games <- game_sales %>% 
  select(Name, Platform, NA_Sales, Critic_Score, Critic_Count) %>%
  arrange(desc(Critic_Score)) %>% 
  head(10)
best_games


# top 10 best games ignoring platform
best_games_grouped <- game_sales %>% 
  group_by(Name) %>% 
  summarise(NA_Sales = sum(NA_Sales), 
            Avg_Critic_Score = mean(Critic_Score), 
            Total_Critics = sum(Critic_Count)) %>% 
  arrange(desc(Avg_Critic_Score)) %>% 
  head(10)
best_games_grouped


# worst games not taking platform into account
worst_games <- game_sales %>% 
  select(Name, Platform, NA_Sales, Critic_Score, Critic_Count) %>%
  arrange(Critic_Score) %>% 
  head(10)
worst_games


# worst games ignoring platform
worst_games_grouped <- game_sales %>% 
  group_by(Name) %>% 
  summarise(NA_Sales = sum(NA_Sales), 
            Avg_Critic_Score = mean(Critic_Score), 
            Total_Critics = sum(Critic_Count)) %>% 
  arrange((Avg_Critic_Score)) %>% 
  head(10)
worst_games_grouped


# correlation between features by rank
corr_matrix <- cor(game_sales[, c(6, 11:14)], method = "spearman")
corr_matrix



############################
# Methods & Models
############################

#############
# Naive Model
#############

# set seed to replicate report results
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1, sample.kind="Rounding")


# split data into training and test sets
test_index <- createDataPartition(y = game_sales$NA_Sales, times = 1, p = 0.2, list = FALSE)


# define training set
train_set <- game_sales[-test_index,]


# define test set
test_set <- game_sales[test_index,]


# average NA sales
mu_hat <- mean(train_set$NA_Sales)
mu_hat


# calculate RMSE for naive model
naive_rmse <- RMSE(test_set$NA_Sales, mu_hat)


# create tibble for calculated RMSE
rmse_model1 <- tibble(Model = "Naive", 
                      Predicted_RMSE = naive_rmse)


# store RMSE for naive model to a results table
rmse_results <- rmse_model1


# print table with Model 1 RMSE result
rmse_results %>% kable() %>% 
  kable_styling(full_width = FALSE) %>%
  row_spec(1, bold = TRUE)


#########################
# Linear Regression Model
#########################


# build linear regression model on training set
lr_model <- train(NA_Sales ~ Critic_Score + User_Score + Platform + Genre + Critic_Count + User_Count,
                  data=train_set,
                  method="lm")


# generate predictions on test set
y_hat <- predict(lr_model, test_set)


# calculate RMSE for linear regression model
lr_rmse<- RMSE(y_hat, test_set$NA_Sales)


# create tibble for calculated RMSE
rmse_model2 <- tibble(Model = "Linear Regression", 
                      Predicted_RMSE = lr_rmse)


# add linear regression results to results table
rmse_results <- bind_rows(rmse_results, 
                          rmse_model2)


# print table of results for all models thus far
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(2, bold = TRUE)



###################
# Elastic Net Model
###################


# Set training control
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              verboseIter = FALSE)


#################################################################
# Note that the below can take a while and use computer resources
#################################################################

# train elastic net model on training data
# warnings expected so suppressed, variables with no variance
enet_model <- suppressWarnings(train(NA_Sales ~ Critic_Score + User_Score + Platform + Genre + Critic_Count + User_Count,
                                     data = train_set,
                                     method = "glmnet",
                                     tuneLength = 25,
                                     trControl = train_control))


# generate predictions on test data
y_hat <- predict(enet_model, test_set)


# calculate RMSE for Elastic Net model
enet_rmse<- RMSE(y_hat, test_set$NA_Sales)


# create tibble for calculated RMSE
rmse_model3 <- tibble(Model = "Elastic Net", 
                      Predicted_RMSE = enet_rmse)


# add elastic net results to results table
rmse_results <- bind_rows(rmse_results, 
                          rmse_model3)


# print table of results for all models thus far
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(3, bold = TRUE)



################
# Random Forest
################


# set model control
control <- trainControl(method="cv", number = 5)


# create grid of values for mtry
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))


########################################################################################################################################
# Note that the below can take a while and use computer resources
########################################################################################################################################

# optimize algorithm
# warnings expected so suppressed, setting the mtry argument of randomForest to a value greater than the number of predictor variables
train_rf <-  suppressWarnings(train(NA_Sales ~ Critic_Score + User_Score + Platform + Genre + Critic_Count + User_Count, 
                   data = train_set,
                   method = "rf", 
                   ntree = 200,
                   trControl = control,
                   tuneGrid = grid))


# plot of results to show how error changes with parameters
ggplot(train_rf)


# get parameter value that is best tune for model
train_rf$bestTune


# fit the model using optimized nodes
fit_rf <- randomForest(NA_Sales ~ Critic_Score + User_Score + Platform + Genre + Critic_Count + User_Count, 
                       data = train_set,
                       minNode = train_rf$bestTune$mtry)


# check if we used enough trees
plot(fit_rf)


# generate predictions on test data
y_hat <- predict(fit_rf, test_set)


# calculate RMSE for Random Forest model
rf_rmse <- RMSE(y_hat, test_set$NA_Sales) 


# create tibble for calculated RMSE
rmse_model4 <- tibble(Model = "Random Forest", 
                      Predicted_RMSE = rf_rmse)


# add random forest results to results table
rmse_results <- bind_rows(rmse_results, 
                          rmse_model4)


# print table of results for all models thus far
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  row_spec(4, bold = TRUE)



###############
# Results
###############


# final results table
rmse_results %>% knitr::kable()%>%
  kable_styling(full_width = FALSE) %>%
  column_spec(2, bold = TRUE)
library(keras)
library(magick)
library(gridExtra)
library(ggplot2)
library(stringr)
library(rsample)
library(caret)
library(doParallel)
library(e1071)

set.seed(1001)

image_dir <- 'C:/Users/ZRose/Desktop/Skewl/breast-histopathology-images/IDC_regular_ps50_idx5/'

imagePatches <- list.files(image_dir, pattern = "\\.png$", full.names = TRUE, recursive = TRUE)

patternZero <- ".*class0\\.png$"
patternOne <- ".*class1\\.png$"

# patterns for positive and negative classes
classZero <- imagePatches[str_detect(imagePatches, patternZero)]
classOne <- imagePatches[str_detect(imagePatches, patternOne)]

# read and process images
proc_images <- function(lowerIndex, upperIndex, imagePatches, classZero, classOne) {
  x <- list()
  y <- c()
  path <- c()
  
  for (img_path in imagePatches[lowerIndex:upperIndex]) {
    full_size_image <- image_read(img_path)
    img_data <- image_data(full_size_image)
    
    if (dim(img_data)[1] == 3 && dim(img_data)[2] == 50 && dim(img_data)[3] == 50) {
      x <- append(x, list(as.integer(img_data)))
      
      if (img_path %in% classZero) {
        y <- c(y, 0)
      } else if (img_path %in% classOne) {
        y <- c(y, 1)
      }
      path <- c(path, img_path)
    }
  }
  
  return(list(x = x, y = y, path = path))
}

# call image proc function with 90000 images
img_list <- proc_images(1,90000, imagePatches, classZero, classOne)

# store images and class / path data in dataframe
df <- data.frame(X = I(img_list$x), y = img_list$y, path = img_list$path)

# split into train and test set
index <- initial_split(df, prop = 0.8)

train <- training(index)
test <- testing(index)

# transform data for NN
train_images <- unlist(train$X) %>% array_reshape(c(length(train$X), 50, 50, 3))
train_images <- train_images / 255
train_labels <- train$y %>% to_categorical(2)

test_images <- unlist(test$X) %>% array_reshape(c(length(test$X), 50, 50, 3))
test_images <- test_images / 255

# diagnostic tool that shows actual and predicted diagnoses and provides img plot and img path
plot <- function(index) {
  img <- as.raster(test$X[[index]]/255)
  sign <- c()
  predSign <- c()
  
  if (test$y[[index]] == 0) {
    sign <- '-'
  }
  else if (test$y[[index]] == 1) {
    sign <- '+'
  }
  
  if (test$pred[[index]] == 0) {
    predSign <- '-'
  }
  else if (test$pred[[index]] == 1) {
    predSign <- '+'
  }
  
  plot0 <- ggplot() +
    annotation_raster(img, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
    ggtitle(paste0("Actual: IDC (", sign, ")", "     Predicted: IDC (", predSign, ")"))
  
  grid.arrange(plot0, ncol = 1)
  print(test$path[[index]])
}

# plot class distribution
ggplot(df, aes(x = as.factor(y))) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(title = "Distribution of Classes in Whole Set",
       x = "Class",
       y = "Count") +
  theme_minimal()

# parallel processing for svm
cl <- makeCluster(8)
registerDoParallel(cl)

train_control <- trainControl(method = "cv", number = 5)
# Tuning grid
tune_grid <- expand.grid(cost = exp(seq(-5,3,len=30)))

# flatten images and transform for svm
train_images_flat <- array_reshape(train_images, c(nrow(train_images), 7500))
test_images_flat <- array_reshape(test_images, c(nrow(test_images), 7500))

train_data <- data.frame(train_images_flat)
train_data$label <- as.factor(train_labels[1:nrow(train_data)])

test_data <- data.frame(test_images_flat)
test_labels <- test$y %>% to_categorical(2)

test_data$label <- as.factor(test_labels[1:nrow(test_data)])

# sample from train and test for svm
train_data_small <- train_data[sample(nrow(train_data), 1000), ]
test_data_small <- test_data[sample(nrow(test_data), 250), ]

# Train the model using linear kernel and tune hyperparameters
sv_caret <- train(label ~ .,
                  data = train_data_small,
                  method = "svmLinear2",
                  trControl = train_control, 
                  preProcess = c("center", "scale"),
                  tuneGrid = tune_grid)
# Final model
sv_final <- svm(label ~ .,
                data = train_data_small,
                type = "C-classification",
                kernel = "linear",
                cost = sv_caret$bestTune$cost)

# predict
predictions <- predict(sv_final, newdata = test_data_small)
confusionMatrix(predictions, test_data_small$label)

stopCluster(cl)

# standard deepnet with 3 dense layers
standardDeepNet <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(50, 50, 3)) %>%
  layer_dense(units = 512) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 256) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# compile with learning rate (adam optimizer)
standardDeepNet %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.000001),
  loss = "binary_crossentropy",
  metrics = c("accuracy") # recall?
)

historyStandard <- standardDeepNet %>%
  fit(train_images, train$y,
      epochs = 8, batch_size = 128,
      validation_split = 0.2)

# predict
predStandard <- predict(standardDeepNet, test_images) %>% `>`(0.5) %>% k_cast("int32")
predLabelsStandard <- as.array(predStandard)

confStandard <- caret::confusionMatrix(data = as.factor(predLabelsStandard),
                                       reference = as.factor(test$y))
confStandard$overall

# convolutional NN with 3 conv_2d layers and 3 dense layers
convnet <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = "relu",
                input_shape = c(50, 50, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

convnet %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001),
  loss = "binary_crossentropy",
  metrics = c("accuracy") # recall?
)

historyConv <- convnet %>%
  fit(train_images, train$y,
      epochs = 8, batch_size = 128,
      validation_split = 0.2)

# predict
predConv <- predict(convnet, test_images) %>% `>`(0.5) %>% k_cast("int32")
predLabels <- as.array(predConv)

confConv <- caret::confusionMatrix(data = as.factor(predLabels),
                               reference = as.factor(test$y))
confConv$overall

# use diagnostic plotting tool
test$pred <- predLabels
plot(100)

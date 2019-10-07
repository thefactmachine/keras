
rm(list = ls())



library(keras)

# could not install this previously as no disk space
# have increase
library("tensorflow")
install_tensorflow(version="nightly")

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

# 8992 ==> each element can be a varied size due to length of each review
length(train_data)

# 2246
length(test_data)


vectorize_sequences <- function(sequences, dimension = 10000)  {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

# 8982 x 10,000
x_train <- vectorize_sequences(train_data)

# 2246 x 10,000
x_test <- vectorize_sequences(test_data)


to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]] + 1] <- 1
  results
}

# 8982 x 46

# can use the keras::to_categorial(train_labels)
one_hot_train_labels <- to_one_hot(train_labels)

# 2246
one_hot_test_labels <- to_one_hot(test_labels)

# there are 46 output values.
# if the intermediate dimensions were significantly less than
# 46 ... then information would be lost....
# the information contained in Ln is dependent on the information
# contained in Ln-x (previous layers)



model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")


model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# set aside 1000 samples
val_indices <- 1:1000
x_val <- x_train[val_indices,]

# 7982 x 10,000
partial_x_train <- x_train[-val_indices,]

y_val <- one_hot_train_labels[val_indices,]

# 7982 x 46
partial_y_train = one_hot_train_labels[-val_indices,]

# lets train for 20 epochs
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

which.max(history$metrics$val_accuracy)

# books says best if after 9 epochs

history_9 <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# Establish a baseline....

test_labels_copy <- test_labels
## test_labels_copy %>% length() ==> 2246
# test_labels_copy %>% head() ===> 3 10 1 4 4 3 ...etc.
# this just jumbles things up....
test_labels_copy <- sample(test_labels_copy)

# about 18% here....
sum(test_labels == test_labels_copy) / length(test_labels)

# =================================
# =================================
# === predictions

#  predictions %>% dim()  ==> 2246 x 46
predictions <- model %>% predict(x_test)

# probablities
predictions[1, ]

































































































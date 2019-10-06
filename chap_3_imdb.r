
rm(list = ls())

library(keras)

imdb <- dataset_imdb(num_words = 10000)

# Using the multi-assignment (%<-%) operator
# train_data <- imdb$train$x
# train_labels <- imdb$train$y
# test_data <- imdb$test$x
# test_labels <- imdb$test$y

c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# 0 stands for negative and 1 stands for postive.
train_labels %>% head()

# length of 25,0000
train_data %>% length()

# just a list of integers
train_data[[1]] %>% head()

# the first is 218 words long
train_data[[1]] %>% length()

# the second is 189 words
train_data[[2]] %>% length()

# get a vector which lists the maximum integer for each set
vct_max_word <- sapply(train_data, max)
# 9999
max(vct_max_word)

# this is how to decode one of the reviews
word_index <- dataset_imdb_word_index()

# names is the word and the value is the actual index
word_index %>% head() %>% names()

# reverses names and index ...the actual value is the word
# and the names is the index
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if(index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}  )

# ==============================================================
# ================================================================
vectorize_sequences <- function(sequences, dimension = 10000)  {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

# both of these have length of 25,000
# so the matrices will have 25,000 x 10,000 ...
# and initially be filled with zeroes....
# train_data[[1]] provides the column numbers for the 1st row
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


# x_train %>% dim() ==> 25000 x 10000

library(keras)
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")





# create validation set

val_indices <- 1:10000
# x_train %>% dim() ==> 25000  x 10000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# x_val %>% dim ==> 10,000 x 10,000 (matrix)
# y_val %>% length ==> 10,000 (vector)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# train the the model for 20 iterations
# with batch size of 512
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# validation_data %>% length() ===> 2
# validation_data[[1]] %>% dim() ===> 10,000 x 10,000
# validation_data[[2]] %>% length() ==> 10,000

# have a look at the history object
str(history)

# use ggplot if available
plot(history)

# to make a custom plot...can call this on history object
as.data.frame(history)

# =================================================
# =================================================
# train using 4 epochs

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
# accuracy of about 87 %
results 

# likelihood of first 10 reviews being positive
model %>% predict(x_test[1:10,])

# =================================================
# =================================================
# further experiments:

# We used two hidden layers....try three
# try 32 hidden units and 64 hidden units
# use mse losss function rather than "binary_crossentropy"
# use the tanh function rather than relu


# following is section 3.5 page 70 (or page 93)







































































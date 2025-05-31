#!/usr/bin/env python
# coding: utf-8

# 
# ## 🐶 End-to-end Multil-class Dog Breed Classification
# 
# This notebook builds an end-to-end multi-class image classifier using TensorFlow 2.x and TensorFlow Hub.
# ###1. Problem
# 
# Identifying the breed of a dog given an image of a dog.
# 
# When I'm sitting at the cafe and I take a photo of a dog, I want to know what breed of dog it is.
# ###2. Data
# 
# The data we're using is from Kaggle's dog breed identification competition.
# 
# https://www.kaggle.com/c/dog-breed-identification/data
# ###3. Evaluation
# 
# The evaluation is a file with prediction probabilities for each dog breed of each test image.
# 
# https://www.kaggle.com/c/dog-breed-identification/overview/evaluation
# ###4. Features
# 
# Some information about the data:
# 
#    * We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.
#    * There are 120 breeds of dogs (this means there are 120 different classes).
#    * There are around 10,000+ images in the training set (these images have labels).
#    * There are around 10,000+ images in the test set (these images have no labels, because we'll want to predict them).
# 
# 

# In[1]:


# Unzip uploaded data into Google Colab
#!unzip "drive/MyDrive/Dog Vision/dog-breed-identification.zip" -d "drive/MyDrive/Dog Vision"


# 
# ##Get our workspace ready
# 
#    * Import TensorFlow 2.x ✅
#    * Import TensorFlow Hub ✅
#    * Make sure we're using a GPU ✅
# 
# 

# In[2]:


#!pip uninstall -y tensorflow tensorflow-hub ml_dtypes

#!pip install tensorflow==2.* tensorflow-hub


# In[3]:


import tensorflow as tf
import tensorflow_hub as hub
print("إصدار TensorFlow:", tf.__version__)
print("إصدار Hub:", hub.__version__)

# Check for GPU availability
print("GPU", "available (Yesssssssssss!!!!!!!)" if tf.config.list_physical_devices("GPU") else "not available:(")


# 
# ##Getting our data ready (turning into Tensors)
# 
# With all machine learning models, our data has to be in numerical format. So that's what we'll be doing first. Turning our images into Tensors (numerical representations).
# 
# Let's start by accessing our data and checking out the labels.
# 

# In[4]:


# Checkout the labels of our data
import pandas as pd
labels_csv = pd.read_csv("drive/MyDrive/Dog Vision/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())


# In[5]:


labels_csv.head()


# In[6]:


# How many images are there of each breed?
labels_csv["breed"].value_counts().plot.bar(figsize=(20,10))


# In[7]:


# What's the median number of images per class?
labels_csv['breed'].value_counts().median()


# In[8]:


# Let's view an image
from IPython.display import Image
Image("drive/MyDrive/Dog Vision/train/16052ac2a6ff7f1fbbc85885d2a7c467.jpg")


# In[9]:


# Create pathnames from image ID's
filenames = ["drive/My Drive/Dog Vision/train/"+ fname+ ".jpg" for fname in labels_csv["id"]]

# Check the first 10
filenames[:10]


# In[10]:


import os
os.listdir("drive/My Drive/Dog Vision/train/")


# In[11]:


# Check whether number of filenames matches number of actual image files
if len(os.listdir("drive/My Drive/Dog Vision/train/")) == len(filenames):
  print("Filenames match actual amount of files!!! Proceed.")
else:
  print("Filenames do not match actual amount of files, checkthe target directory.")


# In[12]:


Image("drive/My Drive/Dog Vision/train/e20e32bf114141e20a1af854ca4d0ecc.jpg")


# In[13]:


labels_csv["breed"][9000]


# In[14]:


Image(filenames[9000])


# 
# 
# Since we've now got our training image filepaths in a list, let's prepare our labels.
# 

# In[15]:


import numpy as np
labels = labels_csv["breed"].to_numpy()
#labels = np.array(labels) # does same thing as above
labels


# In[16]:


len(labels)


# In[17]:


labels.dtype


# In[18]:


# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
  print("Number of labels matches the number of Filenames!!!")
else:
  print("Number of labels does not match the number of Filenames, check data directories")


# In[19]:


# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)


# In[20]:


unique_breeds


# In[21]:


# Turn a single label into an array of boolean
print(labels[0])
labels[0] == unique_breeds


# In[22]:


# Turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]


# In[23]:


# Example : Turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds==labels[0]))
print(boolean_labels[0].argmax())
print(boolean_labels[0].astype(int))


# In[24]:


print(labels[2])
print(np.where(unique_breeds==labels[2]))
print(boolean_labels[2].argmax)
print(boolean_labels[2].astype(int))


# ### Creating our own validation set
# 
# Since the dataset from Kaggle doesn't come with a validation set, we're going to create our own.

# In[25]:


# Setup x & y variables
x = filenames
y = boolean_labels


# In[26]:


len(filenames)


# 
# 
# We're going to start off experimenting with ~1000 images and increase as needed.
# 

# In[27]:


# set number of images to use
NUM_IMAGES = 1000 #@param {type:"slider", min:1000, max:10000, step:1000}
NUM_IMAGES


# In[28]:


# Lets split data into train and validation set
from sklearn.model_selection import train_test_split

# Split them into training and validation of total size NIM_IMAGES
x_train, x_val, y_train, y_val = train_test_split(x[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

len(x_train), len(x_val), len(y_train), len(y_val)


# In[29]:


# Let's have a geez at the training data
x_train[:5], y_train[:2]


# ## Preprocessing Images (turning images into Tensors)
# To preprocess our images into Tensors we're going to write a function which does a few things:
# * 1- Take an image filepath as input
# * 2- Use TensorFlow to read the file and save it to a variable, image
# * 3- Turn our 'image' (a jpg) into Tensors
# * 4- Normalize our image (convert color channel values from from 0-255 to 0-1).
# * 5- Resize the 'image' to be a shape of (224, 224)
# * 6- Return the modified 'image'
# 
# Before we do, let's see what importing an image looks like.

# In[30]:


# Convert image to NumPy array
from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


# In[31]:


image.max(), image.min()


# In[32]:


image[:2]


# In[33]:


# turn image into a tensor
tf.constant(image)[:2]


# Now we've seen what an image looks like as a Tensor, let's make a function to preprocess them.
# 
# ## Preprocessing Images (turning images into Tensors)
# To preprocess our images into Tensors we're going to write a function which does a few things:
# * 1- Take an image filepath as input
# * 2- Use TensorFlow to read the file and save it to a variable, image
# * 3- Turn our 'image' (a jpg) into Tensors
# * 4- Normalize our image (convert color channel values from from 0-255 to 0-1).
# * 5- Resize the 'image' to be a shape of (224, 224)
# * 6- Return the modified 'image'
# 
# Before we do, let's see what importing an image looks like.

# In[34]:


# Define image size
IMG_SIZE = 224

# Create a function for preprocessing images
def process_image(image_path, image_size= IMG_SIZE):
  """
  Takes an image file path and turns the image into a Tensor.
  """
  # Read in an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired value (224, 224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image


# 
# ##Turning our data into batches
# 
# Why turn our data into batches?
# 
# Let's say you're trying to process 10,000+ images in one go... they all might not fit into memory.
# 
# So that's why we do about 32 (this is the batch size) images at a time (you can manually adjust the batch size if need be).
# 
# In order to use TensorFlow effectively, we need our data in the form of Tensor tuples which look like this: (image, label).
# 
# 

# In[35]:


# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path name and the assosciated label,
  processes the image and reutrns a typle of (image, label).
  """
  image = process_image(image_path)
  return image, label


# In[36]:


# Demo of the above
process_image(x[42], tf.constant(y[42]))


# Now we've got a way to turn our data into tuples of Tensors in the form: (image, label), let's make a function to turn all of our data (X & y) into batches!

# In[37]:


# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batch(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have have labels
  if test_data:
    print("Creating test data batch...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths (no labels)
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

  # If the data is a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)), # filepaths
                                              (tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))
    # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)
    data = data.map(get_image_label)
    # Turn the training data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch



# In[38]:


# Create training and validation data batches
train_data=create_data_batch(x_train, y_train)
val_data = create_data_batch(x_val, y_val)


# 
# ## Visualizing Data Batches
# 
# Our data is now in batches, however, these can be a little hard to understand/comprehend, let's visualize them!
# 

# In[39]:


import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(image, label):
  """
  Displays a plot of 25 images and their labels from a data batch.
  """
  # Setup the figure
  plt.figure(figsize=(10, 10))
  # Loop through 25 (for displaying 25 images)
  for i in range(25):
    # Create subplots (5 rows, 5 columns)
    ax = plt.subplot(5, 5, i+1)
    # Display an image
    plt.imshow(image[i])
    # Add the image label as the title
    plt.title(unique_breeds[label[i].argmax()])
    # Turn gird lines off
    plt.axis('off')


# In[40]:


unique_breeds[y[0].argmax()]


# In[41]:


train_data


# In[42]:


train_image, train_label = next(train_data.as_numpy_iterator())
train_image, train_label


# In[43]:


# # Now let's visualize the data in a training batch
len(train_image), len(train_label)


# In[44]:


show_25_images(train_image, train_label)


# In[45]:


# Now let's visualize our validation set
val_image, val_label = next(val_data.as_numpy_iterator())
val_image, val_label


# In[46]:


len(val_image), len(val_label)


# In[47]:


show_25_images(val_image, val_label)


# 
# ## Building a model
# 
# Before we build a model, there are a few things we need to define:
# 
#    * The input shape (our images shape, in the form of Tensors) to our model.
#    * The output shape (image labels, in the form of Tensors) of our model.
#    * The URL of the model we want to use from TensorFlow Hub - https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
# 
# 

# In[48]:


IMG_SIZE


# In[49]:


# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]

# Setup input shape to the model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from TensorFlow Hub
#MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
MODEL_URL = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/2"


# Now we've got our inputs, outputs and model ready to go. Let's put them together into a Keras deep learning model!
# 
# Knowing this, let's create a function which:
# 
#    * Takes the input shape, output shape and the model we've chosen as parameters.
#    * Defines the layers in a Keras model in sequential fashion (do this first, then this, then that).
#    * Compiles the model (says it should be evaluated and improved).
#    * Builds the model (tells the model the input shape it'll be getting).
#     Returns the model.
# 
# All of these steps can be found here: https://www.tensorflow.org/guide/keras/overview
# 

# In[50]:


# Create a function which builds a Keras model
def create_model(input_shape= INPUT_SHAPE, output_shape= OUTPUT_SHAPE, model_url= MODEL_URL):
  print("creating model with:", MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss= tf.keras.losses.CategoricalCrossentropy(),
      optimizer= tf.keras.optimizers.Adam(),
      metrics= ["accuracy"]
  )

  # Build the model
  model.build(INPUT_SHAPE)

  return model


# In[51]:


#!pip install tensorflow==2.15.0 tensorflow-hub==0.15.0


# In[52]:


model = create_model()
model.summary()


# In[53]:


print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)


# ## Creating callbacks
# 
# Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.
# 
# We'll create two callbacks, one for TensorBoard which helps track our models progress and another for early stopping which prevents our model from training for too long.
# 
# ### TensorBoard Callback
# 
# To setup a TensorBoard callback, we need to do 3 things:
# 1. Load the TensorBoard notebook extension ✅
# 2. Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's `fit()` function. ✅
# 3. Visualize our models training logs with the `%tensorboard` magic function (we'll do this after model training).
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard

# In[54]:


# Load TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[55]:


import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
  # Create a log directory for storing TensorBoard logs
  logdir = os.path.join("drive/MyDrive/Dog Vision/logs",
                        # Make it so the logs get tracked whenever we run an experiment
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


# ### Early Stopping Callback
# 
# Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving.
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

# In[56]:


# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)


# ## Training a model (on subset of data)
# 
# Our first model is only going to train on 1000 images, to make sure everything is working.

# In[57]:


NUM_EPOCHS = 100 #@param{type:"slider", min:10, max:100, stip:10}


# In[58]:


# Check to make sure we're still running on a GPU
print("GPU", "available (YESSS!!!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")


# Let's create a function which trains a model.
# 
# * Create a model using `create_model()`
# * Setup a TensorBoard callback using `create_tensorboard_callback()`
# * Call the `fit()` function on our model passing it the training data, validation data, number of epochs to train for (`NUM_EPOCHS`) and the callbacks we'd like to use
# * Return the model

# In[59]:


# Build a function to train and return a trained model
def train_model():
  """
  Trains a given model and returns the trained version.
  """
  # Create a model
  model = create_model()

  # Create new TensorBoard session everytime we train a model
  tensorboard = create_tensorboard_callback()

  # Fit the model to the data passing it the callbacks we created
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, early_stopping])
  # Return the fitted model
  return model


# In[60]:


# Fit the model to the data
model = train_model()


# **Question:** It looks like our model is overfitting because it's performing far better on the training dataset than the validation dataset, what are some ways to prevent model overfitting in deep learning neural networks?
# 
# **Note:** Overfitting to begin with is a good thing! It means our model is learning!!!

# 

# ### Checking the TensorBoard logs
# 
# The TensorBoard magic function (`%tensorboard`) will access the logs directory we created earlier and visualize its contents.

# In[61]:


tensorboard --logdir drive/MyDrive/Dog\ Vision/logs


# ## Making and evaluating predictions using a trained model

# In[62]:


val_data


# In[63]:


# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1)
predictions


# In[64]:


predictions[0]


# In[65]:


# First prediction
index = 42
print(predictions[index])
print(f"Max value (probability of prediction): {np.max(predictions[index])}")
print(f"Sum: {np.sum(predictions[index])}")
print(f"Max index: {np.argmax(predictions[index])}")
print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")


# In[66]:


unique_breeds[113]


# 
# Having the the above functionality is great but we want to be able to do it at
# 
# *   List item
# *   List item scale.
# 
# And it would be even better if we could see the image the prediction is being made on!
# 
# **Note:** Prediction probabilities are also known as confidence levels.

# In[67]:


# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[81])
pred_label


# Now since our validation data is still in a batch dataset, we'll have to unbatchify it to make predictions on the validation images and then compare those predictions to the validation labels (truth labels).

# In[68]:


# Create a function to unbatch a batch dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and reutrns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in val_data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels

# Unbatchify the validation data
val_images, val_labels = unbatchify(val_data)
val_images[81], val_labels[81]


# Now we've got ways to get get:
# * Prediction labels
# * Validation labels (truth labels)
# * Validation images
# 
# Let's make some function to make these all a bit more visaulize.
# 
# We'll create a function which:
# * Takes an array of prediction probabilities, an array of truth labels and an array of images and an integer. ✅
# * Convert the prediction probabilities to a predicted label. ✅
# * Plot the predicted label, its predicted probability, the truth label and the target image on a single plot. ✅

# In[69]:


import matplotlib.pyplot as plt


# In[70]:


def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth and image for sample n
  """
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

  # Get the pred label
  pred_label = get_pred_label(pred_prob)

  # Plot image & remove ticks
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  if pred_label == true_label:
   color = "green"
  else:
   color = "red"

  # Change plot title to be predicted, probability of prediction and truth label
  plt.title("{} {:2.0f}% {}".format(pred_label,
                                    np.max(pred_prob)*100,
                                    true_label),
                                    color=color)


# In[71]:


plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,
          n=110)


# Now we've got one function to visualize our models top prediction, let's make another to view our models top 10 predictions.
# 
# This function will:
# * Take an input of prediction probabilities array and a ground truth array and an integer ✅
# * Find the prediction using `get_pred_label()` ✅
# * Find the top 10:
#   * Prediction probabilities indexes ✅
#   * Prediction probabilities values ✅
#   * Prediction labels ✅
# * Plot the top 10 prediction probability values and labels, coloring the true label green ✅

# In[72]:


def plot_pred_conf(prediction_probabilities, labels, n=1):
  """
  Plus the top 10 highest prediction confidences along with the truth label for sample n.
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  # Get the predicted label
  pred_label = get_pred_label(pred_prob)

  # Find the top 10 prediction confidence indexes
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  # Find the top 10 prediction confidence values
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  # Find the top 10 prediction labels
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                     top_10_pred_values,
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  # Change color of true label
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass


# In[73]:


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)


# Now we've got some function to help us visualize our predictions and evaluate our modle, let's check out a few.

# In[74]:


# Let's check out a few predictions and their different values
i_multiplier = 20
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(10*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                 labels=val_labels,
                 n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


# ## Saving and reloading a trained model

# In[75]:


# Create a function to save a model
def save_model(model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (string).
  """
  # Create a model directory pathname with current time
  modeldir = os.path.join("drive/MyDrive/Dog Vision/models",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path = modeldir + "-" + suffix + ".h5" # save format of model
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path


# In[76]:


# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


# Now we've got functions to save and load a trained model, let's make sure they work!

# In[77]:


# Save our model trained on 1000 images
save_model(model, suffix="1000-images-mobilenetv2-Adam")


# In[78]:


# Load a trained model
loaded_1000_image_model = load_model('drive/MyDrive/Dog Vision/models/20250525-12381748176734-1000-images-mobilenetv2-Adam.h5')


# In[79]:


# Evaluate the pre-saved model
model.evaluate(val_data)


# In[80]:


# Evaluate the loaded model
loaded_1000_image_model.evaluate(val_data)


# ## Training a big dog model 🐶 (on the full data)

# In[81]:


full_data = create_data_batch(x,y)


# In[82]:


full_data


# In[83]:


# Create a model for full model
full_model = create_model()


# In[84]:


# Create full model callbacks
full_model_tensorboard = create_tensorboard_callback()
# No validation set when training on all the data, so we can't monitor validation accuracy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=3)


# In[85]:


# Fit the full model to the full data
#full_model.fit(x=full_data,
#               epochs=NUM_EPOCHS,
#               callbacks=[full_model_tensorboard, full_model_early_stopping])


# In[86]:


save_model(full_model, suffix="full-data-mobilenetv2-Adam")


# In[87]:


load_full_model= load_model("drive/MyDrive/Dog Vision/models/20250526-15371748273829-full-data-mobilenetv2-Adam.h5")


# ## Making predictions on the test dataset
# 
# Since our model has been trained on images in the form of Tensor batches, to make predictions on the test data, we'll have to get it into the same format.
# 
# Luckily we created `create_data_batches()` earlier which can take a list of filenames as input and conver them into Tensor batches.
# 
# To make predictions on the test data, we'll:
# * Get the test image filenames. ✅
# * Convert the filenames into test data batches using `create_data_batches()` and setting the `test_data` parameter to `True` (since the test data doesn't have labels). ✅
# * Make a predictions array by passing the test batches to the `predict()` method called on our model.

# In[88]:


# Load test image filenames
test_path = "drive/MyDrive/Dog Vision/test/"
test_filename = [test_path + fname for fname in os.listdir(test_path)]
test_filename[:10]


# In[89]:


len(test_filename)


# In[90]:


# Create test data batch
test_data = create_data_batch(test_filename, test_data=True)


# In[91]:


test_data


# **Note:** Calling `predict()` on our full model and passing it the test data batch will take a long time to run (about a ~1hr). This is because we have to process ~10,000+ images and get our model to find patterns in those images and generate predictions based on what its learned in the training dataset.

# In[93]:


# Make predictions on test data batch using the loaded full model
test_prediction = loaded_full_model.predict(test_data,
                                              verbose=1)
test_prediction


# In[ ]:


# Save predictions (NumPy array) to csv file (for access later)
np.savetxt("drive/MyDrive/Dog Vision/preds_array.csv", test_predction, delimiter=",")


# In[ ]:


# Load predictions (NumPy array) from csv file
test_predictions = np.loadtxt("drive/MyDrive/Dog Vision/preds_array.csv", delimiter=",")


# In[ ]:


test_predictions[:10]


# In[ ]:


test_predictions.shape


# ## Preparing test dataset predictions for Kaggle
# 
# Looking at the Kaggle sample submission, we find that it wants our models prediction probaiblity outputs in a DataFrame with an ID and a column for each different dog breed.
# https://www.kaggle.com/c/dog-breed-identification/overview/evaluation
# 
# To get the data in this format, we'll:
# * Create a pandas DataFrame with an ID column as well as a column for each dog breed. ✅
# * Add data to the ID column by extracting the test image ID's from their filepaths.
# * Add data (the prediction probabilites) to each of the dog breed columns.
# * Export the DataFrame as a CSV to submit it to Kaggle.

# In[ ]:


# ["id"] + list(unique_breeds)


# In[ ]:


# Create a pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()


# In[ ]:


# Append test image ID's to predictions DataFrame
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids


# In[ ]:


preds_df.head()


# In[ ]:


# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()


# In[ ]:


# Save our predictions dataframe to CSV for submission to Kaggle
preds_df.to_csv("drive/My Drive/Dog Vision/full_model_predictions_submission_1_mobilenetV2.csv",
                index=False)


# ## Making predictions on custom images
# 
# To make predictions on custom images, we'll:
# * Get the filepaths of our own images.
# * Turn the filepaths into data batches using `create_data_batches()`. And since our custom images won't have labels, we set the `test_data` parameter to `True`.
# * Pass the custom image data batch to our model's `predict()` method.
# * Convert the prediction output probabilities to predictions labels.
# * Compare the predicted labels to the custom images.

# In[ ]:


# Get custom image filepaths
custom_path = "drive/My Drive/Dog Vision/my-dog-photos/"
custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]


# In[ ]:


custom_image_paths


# In[ ]:


# Turn custom images into batch datasets
custom_data = create_data_batches(custom_image_paths, test_data=True)
custom_data


# In[ ]:


# Make predictions on the custom data
custom_preds = loaded_full_model.predict(custom_data)


# In[ ]:


custom_preds.shape


# In[ ]:


# Get custom image prediction labels
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels


# In[ ]:


# Get custom images (our unbatchify() function won't work since there aren't labels... maybe we could fix this later)
custom_images = []
# Loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
  custom_images.append(image)


# In[ ]:


# Check custom image predictions
plt.figure(figsize=(10, 10))
for i, image in enumerate(custom_images):
  plt.subplot(1, 3, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(custom_pred_labels[i])
  plt.imshow(image)


# In[ ]:





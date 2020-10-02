Classifying Radio Signals from Space is divided into following tasks:

The data set is available in images, the actual radio signal (time series data) is trainsformed to 2D spectrogram images & the data set is downloaded from the seti.org site. 
There are four classes of signals in the dataset:
  1. Squiggle
  2. Narrowband
  3. Narrowband DRD
  4. Noise

Task 1:
  Import essential modules and helper functions from NumPy, Matplotlib, and Keras.
Task 2: Load and Preprocess SETI Data
  Display 2D spectrograms using Matplotlib.
  Reshape the input data with NumPy.
Task 3: Create Training and Validation Data Generators
  Generate batches of tensor image data with real-time data augmentation.
  Specify paths to training and validation image directories and generates batches of augmented data.
Task 4: Create a Convolutional Neural Network (CNN) Model
  Design a convolutional neural network with 2 convolution layers and 1 fully connected layers to predict four signal types.
  Use Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.
Task 5: Learning Rate Scheduling and Compile the Model
  When training a model, it is often recommended to lower the learning rate as the training progresses.
  Apply an exponential decay function to the provided initial learning rate.
Task 6: Train the Model
  Train the CNN by invoking the model.fit() method.
  Use ModelCheckpoint() to save the weights associated with the higher validation accuracy after every epoch
  Display live training loss and accuracy plots in Jupyter Notebook using livelossplot.
Task 7: Evaluate the Model
  Evaluate the CNN by invoking the model.fit() method.
  Obtain the classification report to note the precision and recall of your classifier.

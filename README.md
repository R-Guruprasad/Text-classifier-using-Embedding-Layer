# Text-classifier-using-Embedding-Layer
## AIM
To create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D.

## PROBLEM STATEMENT AND DATASET
The program enables us to classify the given BBC dataset into its respective areas like different categories, for example buisness, sports and tech using Deep learning techniques, which includes loading and preprocessing the data, creating the neural network model, training and evaluation its performance.


## DESIGN STEPS
## STEP 1:
Unzip the zip file and load the BBC news dataset, split it into training and validation dataset.

## STEP 2:
Implement a function to convert the text into lower cases, remove the stop words and eliminate punctuation.

## STEP 3:
Create a TextVectorizer layer to tokenize and convert the dataset into sequences for model training.

##  STEP 4:
Use TensorFlow's StringLookup layer to encode text labels into numerical format.

## STEP 5:
Use TensorFlow's StringLookup layer to encode text labels into numerical format.

## STEP 6:
Train the model for 30 epochs using the prepared training data and validate its performance on the validation set.

## STEP 7:
Evaluate the model's accuracy and loss, and plot the results to track performance over time.

## PROGRAM
```py
#Import necessary libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
```py
#To unzip and read the csv file inside the zip file

import zipfile

with zipfile.ZipFile('/content/BBC News Train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_data')
```
```py
with open("/content/BBC News Train.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"The second line (first data point) looks like this:\n\n{csvfile.readline()}")
```
```py
  # Define the global variables
VOCAB_SIZE =1000
EMBEDDING_DIM =16
MAX_LENGTH =120
TRAINING_SPLIT = 0.8
```
```py
data_dir = "/content/BBC News Train.csv"
data = np.loadtxt(data_dir, delimiter=',', skiprows=1, dtype='str', comments=None)
print(f"Shape of the data: {data.shape}")
print(f"{data[0]}\n{data[1]}")
```
```py
# Test the function
print(f"There are {len(data)} sentence-label pairs in the dataset.\n")
print(f"First sentence has {len((data[0,1]).split())} words.\n")
print(f"The first 5 labels are {data[:5,2]}")
```
```py
# train_val_datasets
def train_val_datasets(data,train_split=0.8):
    '''
    Splits data into traning and validations sets

    Args:
        data (np.array): array with two columns, first one is the label, the second is the text

    Returns:
        (tf.data.Dataset, tf.data.Dataset): tuple containing the train and validation datasets
    '''
   ### START CODE HERE ###

    # Compute the number of samples that will be used for training
    train_size =int(len(data)*train_split)

    # Slice the dataset to get only the texts and labels
    texts =data[:,1]
    labels =data[:,2]
    
    # Split the texts and labels into train/validation splits
    train_texts =texts[:train_size]
    validation_texts =texts[train_size:]
    train_labels =labels[:train_size]
    validation_labels =labels[train_size:]

    # Create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset =tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))

    ### END CODE HERE ###
    return train_dataset, validation_dataset,
```
```py
# Create the datasets
train_dataset, validation_dataset = train_val_datasets(data)
print('Name: R Guruprasad Register Number: 212222240033   ')
print(f"There are {train_dataset.cardinality()} sentence-label pairs for training.\n")
print(f"There are {validation_dataset.cardinality()} sentence-label pairs for validation.\n")
```
```py
def standardize_func(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (tf.string): sentence to remove the stopwords from

    Returns:
        sentence (tf.string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",  "hers", "herself", "him", "himself", "his", "how",  "i", "if", "in", "into", "is", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",  "should", "so", "some", "such", "than", "that",  "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",  "were", "what",  "when", "where", "which", "while", "who", "whom", "why", "why", "with", "would", "you",  "your", "yours", "yourself", "yourselves", "'m",  "'d", "'ll", "'re", "'ve", "'s", "'d"]

    # Sentence converted to lowercase-only
    sentence = tf.strings.lower(sentence)

    # Remove stopwords
    for word in stopwords:
        if word[0] == "'":
            sentence = tf.strings.regex_replace(sentence, rf"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")

    # Remove punctuation
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', "")


    return sentence
```
```py
# GRADED FUNCTION: fit_vectorizer
def fit_vectorizer(train_sentences, standardize_func):
    '''
    Defines and adapts the text vectorizer

    Args:
        train_sentences (tf.data.Dataset): sentences from the train dataset to fit the TextVectorization layer
        standardize_func (FunctionType): function to remove stopwords and punctuation, and lowercase texts.
    Returns:
        TextVectorization: adapted instance of TextVectorization layer
    '''
    ### START CODE HERE ###

    # If train_sentences is a NumPy array, convert it to a TensorFlow Dataset


    # Initialize the TextVectorization layer
    vectorizer =tf.keras.layers.TextVectorization(
        standardize=standardize_func,
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH
    )


    # Adapt the vectorizer to the training sentences
    vectorizer.adapt(train_sentences)

    ### END CODE HERE ###

    return vectorizer
```
```py
# Create the vectorizer
text_only_dataset = train_dataset.map(lambda text, label: text)
vectorizer = fit_vectorizer(text_only_dataset, standardize_func)
vocab_size = vectorizer.vocabulary_size()
print('Name: R Guruprasad Register Number: 212222240033  ')
print(f"Vocabulary contains {vocab_size} words\n")
```
```py
# GRADED FUNCTION: fit_label_encoder
def fit_label_encoder(train_labels, validation_labels):
    """Creates an instance of a StringLookup, and trains it on all labels

    Args:
        train_labels (tf.data.Dataset): dataset of train labels
        validation_labels (tf.data.Dataset): dataset of validation labels

    Returns:
        tf.keras.layers.StringLookup: adapted encoder for train and validation labels
    """
    ### START CODE HERE ###

    # Join the two label datasets by concatenating them
    labels=train_labels.concatenate(validation_labels)

    # Instantiate the StringLookup layer. We set mask_token=None and num_oov_indices=0 to avoid OOV tokens
    label_encoder=tf.keras.layers.StringLookup(mask_token=None,num_oov_indices=0)

    # Fit the StringLookup layer on the concatenated labels
    label_encoder.adapt(labels)

    ### END CODE HERE ###

    return label_encoder
```
```py
# Create the label encoder
train_labels_only = train_dataset.map(lambda text, label: label)
validation_labels_only = validation_dataset.map(lambda text, label: label)

label_encoder = fit_label_encoder(train_labels_only,validation_labels_only)
print('Name:R Guruprasad Register Number: 212222240033')
print(f'Unique labels: {label_encoder.get_vocabulary()}')
```
```py
# GRADED FUNCTION: preprocess_dataset
def preprocess_dataset(dataset, text_vectorizer, label_encoder):
    """Apply the preprocessing to a dataset

    Args:
        dataset (tf.data.Dataset): dataset to preprocess
        text_vectorizer (tf.keras.layers.TextVectorization ): text vectorizer
        label_encoder (tf.keras.layers.StringLookup): label encoder

    Returns:
        tf.data.Dataset: transformed dataset
    """

      ### START CODE HERE ###

    # Apply text vectorization and label encoding
    dataset = dataset.map(lambda x, y: (text_vectorizer(x), label_encoder(y))) 
    dataset = dataset.batch(32) # Set a batchsize of 32
    # Set the batch size to 32


    ### END CODE HERE ###

    return dataset
```
```py
# Preprocess your dataset
train_proc_dataset = preprocess_dataset(train_dataset, vectorizer, label_encoder)
validation_proc_dataset = preprocess_dataset(validation_dataset, vectorizer, label_encoder)
```
```py
train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())
print('Name: R Guruprasad Register Number:212222240033')
print(f"Shape of the train batch: {train_batch[0].shape}")
print(f"Shape of the validation batch: {validation_batch[0].shape}")
```
```py
# GRADED FUNCTION: create_model
def create_model():
    """
    Creates a text classifier model
    Returns:
      tf.keras Model: the text classifier model
    """

      ### START CODE HERE ###

    model =tf.keras.Sequential([
        #tf.keras.Input(shape=(120,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE,output_dim=EMBEDDING_DIM,input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(5,activation='softmax')

    ])


    # Compile the model with appropriate loss, optimizer, and metrics
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



    ### END CODE HERE ###

    return model

# Get the untrained model
model =create_model()


example_batch = train_proc_dataset.take(1)

try:
	model.evaluate(example_batch, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
	predictions = model.predict(example_batch, verbose=False)
	print(f"predictions have shape: {predictions.shape}")


  history = model.fit(train_proc_dataset, epochs=30, validation_data=validation_proc_dataset)
```
```py
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
print('Name:        Register Number:       ')
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```
### Name: R Guruprasad
### Register Number: 212222240033

## OUTPUT

![image](https://github.com/user-attachments/assets/2af99f27-38a0-42f9-ab03-453fc9ae2ae3)

![image](https://github.com/user-attachments/assets/cfaf3802-1117-4f9d-b3ca-e63cc7ea1db4)

![image](https://github.com/user-attachments/assets/1d9ea53a-9fcd-43f7-a812-12e910847de0)

![image](https://github.com/user-attachments/assets/60909e8a-074c-4b08-89c6-bacbe953d7c1)

![image](https://github.com/user-attachments/assets/069e85fa-b8af-4edb-92f9-e25a7e1a1d5f)

![image](https://github.com/user-attachments/assets/12e4636f-22d5-47ea-bfd0-fe38542e3afe)

![image](https://github.com/user-attachments/assets/27951e74-210a-46e0-8f1f-674f94ede238)

![image](https://github.com/user-attachments/assets/37f09511-3ba3-45f3-8652-5936f347f857)

### Loss, Accuracy Vs Iteration Plot

![image](https://github.com/user-attachments/assets/38d0415f-209c-478a-a46e-a2a43fa7a68e)

![image](https://github.com/user-attachments/assets/9ecd5bd6-1c38-4bc3-b5cc-b478016567e9)

## RESULT
Thus the program to create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D is implemented successfully.


# RNN_Movies_Review
To review the IMDB dataset and analyse the review as positive - 1 or negative - 0

Dataset: from tensorflow - imdb
  50K datapoints, vocab size = 10K, Feature dimensions for Embedding layer = 128

  Feature Engineering: Add an Embedding Layer to convert the reviews to vectors.
        Default Reviews are OHE , add padded sequences to make all the reviews of even length(500)

  Simple RNN and Embedding Layer: First the encoded review is sent to the Embedding layer(Sequential,embedding) with 128 dimensions, 10k  vocab size and 500 as input length(zero padded review).
  Then a SimpleRNN layer is added to that, then a dense layer for output with sigmoid activation function(Binary sentiment=+ or -)
  Dropout and recurrent dropout are added to generalize the model(prevent overfitting) and improve the overall accuracy.

  Prediction: A user input in the form of review is taken, and the imdb get word index is used to get the index(add 3 to the original), 
            Do Zero padding for the OHE of indices obtained.
            Use the model and predict the sentiment using the encoded review,
            >0.5 = Positive, else Negative

  Streamlit WebApp: Get the user input review and display the prediction

  Deployment: Streamlit cloud

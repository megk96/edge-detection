text-mining.py

The text-mining.py consists of 5 different methods.

main()
Starts the timer on the running of the program using the time library
The data is read as a Pandas dataframe
The commented out code is to check the accuracy of the data before cleaning it for Question 4
question_one(), question_two(), question_three(), and question_four() are called in sequence.

DATA INSPECTION
question_one(df)
Dataframe is received as input
Unique values are found for the column value Sentiment.
Value_counts is used to find the number of tweets for each sentiment and it is ordered decreasingly.
Tweets are first filtered by sentiment = Extremely Positive and then grouped by the date to find the date with the maximum positive tweets.
Preprocessing techniques, string functions such as lower() and replace() are used.
Replace uses regex to find non alphabetical characters and replace them with a whitespace.
The formatted Dataframe is returned


UTILITY FUNCTION
get_stats(series)
Series is received as input.
The FreqDist function is used on the entire corpus.
It outputs a dictionary with terms and their frequencies.
This is used to calculate distinct words - keys of dictionary of frequency distribution
Total words = total number of instances in the frequency distribution


DATA CLEANING
question_two(df)
The formatted Dataframe is taken as input.
The nltk library provides a list of English stopwords, the stopwords are removed.
The split() function is used to split and proved to be more time efficient that word_tokenize
Lambda functions are used to remove stop words and words less than 2 characters
get_stats() is called to get the requried statistics.


HISTOGRAM AND DOCUMENT FREQUENCY
question_three(df)
The Counter function from collections is used
The word frequencies are first found
Using the word frequencies, the document frequencies are found.
It is stored as a list dividing by document size to get the fraction of documents
This is used to plot a line graph using matplotlib

MULTINOMIAL NAIVE BAYES
question_four(df)
For running it before data is cleaned, we have tokenized=False
For running it after data is cleaned, we have tokenized=True
The tokens are converted to string and then to numpy array
This is fed to CountVectorizer
This input is then pushed to Multinomial Naive Bayes classifier
The classifier is fit on the training data only
Data is not traditionally split and error rate is calculated on the training data

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

image-processing.py

The image-processing.py file consists of 4 different methods one for each sub-part of the question.

GRAYSCALE AND BINARY
question_one()
Reads the input as color
Outputs the shape
Uses the skimage rgb2gray for conversion.
A threshold on the grayscale is used for Binary.
The pixels are already normalized from 0 to 1 so a 0.5 threshold for black and white works perfectly.

NOISE AND FILTERS
question_two()
All functions are imported from skimage.util and filters from scipy.ndimage
The image is first perturbed by random gaussian noise of var=0.1 using random_noise
The perturbed image is then passed through a gaussian filter using gaussian_filter
The perturbed image is then passed through a uniform filter
To further experiment the gaussian filtered image is then passed through a uniform filter

K-MEANS SEGMENTATION
Compactness is the control parameter
15 gives the most sensible results.
slic function is used from the skimage.segmentation library
To visually represent the results, mark_boundaries is used to show superimposed image


CANNY EDGE DETECTION AND HOUGH TRANSFORM
Gaussian filter is first applied to remove the noise, different values of sigma are experimented before settling on this value - 0.55
canny method is used to perform the edge detection
This image is then passed through probabilistic hough line
After obtaining the Hough Lines, they are plotted using matplotlib

----------------------------------------------------------------------------------------------------------------------------------------------------
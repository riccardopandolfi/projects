# Web and Social Media Analytics Assignment

This project is an assignment for the Web and Social Media Analytics course. It involves analyzing tweets related to the 2020 U.S. Presidential Election and performing sentiment analysis using machine learning techniques.

## Data

The project uses two datasets: `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv`. These datasets contain tweets related to Donald Trump and Joe Biden, respectively.

## Packages

The following packages are used in this project:
- numpy
- pandas
- plotly
- matplotlib
- seaborn
- nltk
- gensim
- wordcloud
- sklearn

## Data Cleaning

The data cleaning process involves the following steps:
1. Importing the datasets
2. Deleting rows with missing values
3. Standardizing the country name to "United States"
4. Removing special characters, punctuation, URLs, and stop words from the tweets
5. Lemmatizing the cleaned tweets
6. Performing sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment intensity analyzer

## Exploratory Data Analysis

Several exploratory data analysis (EDA) visualizations are performed on the cleaned data. Here are some of the plots generated:

1. Number of Tweets by Candidate: A bar chart showing the number of tweets for each candidate (Donald Trump and Joe Biden).
2. Number of Tweets by State: A bar chart showing the number of tweets for each candidate in different U.S. states.
3. Number of Tweets by Day: A line chart showing the number of tweets per day for each candidate.
4. Positive Joe Biden Tweets by State: A choropleth map showing the distribution of positive tweets about Joe Biden across U.S. states.
5. Number of Positive and Negative Tweets Before the Election: A bar chart showing the number of positive and negative tweets for each candidate before the election.
6. Sentiment Change Over Time: A line chart showing the sentiment scores over time for each candidate.

## Word Clouds

Word clouds are generated to visualize the most frequent words in the tweets. Separate word clouds are created for tweets related to Donald Trump and Joe Biden.

## Word Similarity

Word similarity analysis is performed using Word2Vec, a library for text analysis. The most similar words to "joebiden" and "donaldtrump" are identified and visualized in bar charts.

## Topic Modeling - LDA

Topic modeling using Latent Dirichlet Allocation (LDA) is performed on the tweets and users' descriptions. The top words for each topic are extracted and displayed.

## Classifier

A sentiment classifier is built using logistic regression. The tweets are vectorized using TF-IDF, and the model is trained and evaluated using accuracy, precision, recall, and F1-score. Grid search is used to tune the hyperparameters of the logistic regression model.

## Model Evaluation

The trained logistic regression model is evaluated using several performance metrics and visualizations:

1. Confusion Matrix: A heatmap representing the confusion matrix of the model's predictions.
2. ROC Curve: A plot showing the receiver operating characteristic (ROC) curve of the model.
3. Precision-Recall Curve: A plot showing the precision-recall curve of the model.
4. Feature Importances: A bar chart showing the top 20 important features (words) for predicting the sentiment.

## Usage

To run the code, follow these steps:
1. Install the required packages mentioned in the "Packages" section.
2. Place the datasets (`hashtag_donaldtrump.csv` and `hashtag_joebiden.csv`) in the same directory as the code file.
3. Run the code in your preferred Python environment or IDE. Make sure to set the working directory to the location where the code file and datasets are located.
4. The code will execute the data cleaning process, perform exploratory data analysis, generate word clouds, conduct word similarity analysis, apply topic modeling using LDA, build and evaluate the sentiment classifier, and generate performance visualizations.
5. You can customize the code according to your requirements, such as modifying the data cleaning steps, adding new visualizations, or experimenting with different machine learning models.

Note: Before running the code, ensure that you have obtained the necessary API keys or credentials for accessing Twitter data, if required. If you encounter any errors or issues, please refer to the documentation of the packages used or consult the course materials for assistance.

Enjoy exploring and analyzing the tweets related to the 2020 U.S. Presidential Election!

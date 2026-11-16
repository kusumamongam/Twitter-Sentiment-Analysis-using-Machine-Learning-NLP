# Twitter Sentiment Analysis Using Machine Learning
A three-class text classification project that analyzes tweet content and predicts whether the expressed sentiment is positive, negative, or neutral. 
The project demonstrates an end-to-end
machine learning workflow, from exploratory data analysis and NLP
preprocessing to TF-IDF feature extraction, model comparison,
evaluation, and prediction on new text.

## Project Overview
Social media platforms contain large volumes of user-generated text.
Understanding the sentiment expressed in these posts can help
organizations monitor public opinion, identify customer concerns, and
understand reactions to products, services, or events.

This project builds and evaluates machine learning models to classify
tweets into three sentiment categories:
- Positive
- Negative
- Neutral

Four classification algorithms are compared using a held-out test set.
The best model is selected using weighted F1-score, and the selected
model is then used for final evaluation and custom tweet predictions.

## Key Results
The notebook was executed on a dataset containing 27,481 records.
One row with missing tweet text was removed, leaving 27,480 records
for analysis. After preprocessing, the data was split into training and
testing sets using an 80:20 ratio.

### Best-performing model: Random Forest

  Evaluation metric      Result
  Accuracy               70.50%
Weighted precision     70.83%
Weighted recall        70.50%
Weighted F1-score      70.46%

## Model Comparison
#### Results from the executed notebook:
  Model              Accuracy       Weighted       Weighted       Weighted
Precision         Recall       F1-score
  Random               70.50%         70.83%         70.50%     70.46%
Forest                                                    

  Support              70.00%         71.69%         70.00%         69.91%
Vector
Machine
(SVC)                                                     

  Logistic             69.46%         70.33%         69.46%         69.50%
Regression                                                
Gradient             64.99%         68.74%         64.99%         64.26%
Boosting
Metrics are from this notebook execution and may vary slightly if the
environment, library versions, or dataset changes.
Dataset
The notebook expects a CSV file named tweets.csv in the project
directory. The dataset used in the executed notebook contains 27,481
rows and 4 columns:
  Column                              Description
  textID                            Identifier associated with a tweet

  text                              Original tweet text used as the
model input

  selected_text                     Selected text associated with the
sentiment annotation; not used as a
model feature
sentiment                         Target label: positive, negative,
or neutral
Sentiment distribution after removing the missing text row
  Sentiment     Number of tweets
  Neutral                 11,117
Positive                 8,582
Negative                 7,781
Total           27,480

The dataset is not included in this repository. Add the permitted
dataset file locally as tweets.csv before running the notebook. Follow
the dataset provider's terms and licensing requirements.
Project Workflow
1. Data loading and validation --- Load tweets.csv, check
   required columns, inspect the records, and remove rows with missing
   text or sentiment labels.
2. Exploratory Data Analysis (EDA) --- Examine sentiment counts and
   proportions and generate word clouds for each sentiment category.
3. Text preprocessing --- Remove HTML tags and URLs, convert text
   to lowercase, tokenize, remove English stopwords and non-alphabetic
   tokens, and apply Porter stemming.
4. Train-test split --- Split the cleaned text and labels into 80%
   training and 20% testing data, using stratification to preserve
   class proportions.
5. TF-IDF feature extraction --- Learn the vocabulary and inverse
   document frequency values from the training data only; transform the
   test data using the fitted vectorizer.
6. Model training and comparison --- Train Logistic Regression,
   Support Vector Machine (SVC), Random Forest, and Gradient Boosting
   classifiers.
7. Model selection --- Select the model with the highest weighted
   F1-score.
8. Model evaluation --- Review the classification report and
   confusion matrix for the selected model.
9. Custom predictions --- Apply the same preprocessing and fitted
   TF-IDF transformation to new tweets, then predict their sentiment
   using the selected model.
Data Leakage Prevention
A key improvement in this workflow is that the train-test split is
performed before fitting TF-IDF.
- fit_transform() is applied only to the training text.
- transform() is applied to the test text using the already-fitted
  vectorizer.
- The test set is therefore not used to learn the TF-IDF vocabulary or
  inverse document frequency values.
This helps ensure that the test data remains unseen during feature
learning and provides a more reliable estimate of model performance.
Model Evaluation
The models are compared using:
- Accuracy --- Overall proportion of correctly classified tweets.
- Precision (weighted) --- Weighted measure of how many predicted
  labels are correct.
- Recall (weighted) --- Weighted measure of how many actual
  instances are identified.
- F1-score (weighted) --- Weighted harmonic mean of precision and
  recall, accounting for the number of samples in each class.
Weighted F1-score is used for model selection. In the executed notebook,
Random Forest achieved the highest weighted F1-score (0.7046) among
the four tested models.
Random Forest: class-wise test results
  Class        Precision   Recall   F1-score   Support
  Negative          0.73     0.62       0.67     1,555
Neutral           0.66     0.73       0.69     2,204
Positive          0.76     0.75       0.75     1,716

The confusion matrix and classification report are generated in the
notebook to support a more detailed review of model performance.
Example Predictions
The notebook tests the selected model on three new tweets:

  Input tweet                          Predicted sentiment
  I hate Twitter                     Negative
The weather is awesome today!      Positive
I am using Twitter this morning.   Neutral

These examples demonstrate how the trained workflow can classify new
text. They are illustrative predictions, not a substitute for broader
validation.
Technologies and Libraries
- Language: Python
- Environment: Jupyter Notebook
- Data handling: pandas, NumPy
- Visualization: Matplotlib, Seaborn, WordCloud
- NLP preprocessing: NLTK
- Feature extraction: scikit-learn TF-IDF Vectorizer
- Machine learning: scikit-learn
- Evaluation: Accuracy, weighted precision, weighted recall,
  weighted F1-score, classification report, confusion matrix

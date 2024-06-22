
# NLP Functions and Topic Tagging

## Overview

This project provides a set of functions for natural language processing (NLP) tasks such as text preprocessing and topic tagging using Latent Dirichlet Allocation (LDA). These functions aim to clean and analyze text data to extract meaningful topics or tags that can be associated with input questions or texts.

## Dependencies

Ensure you have the following Python packages installed:

- BeautifulSoup (`bs4`)
- NLTK (`nltk`)
- scikit-learn (`sklearn`)
- NumPy (`numpy`)

Install NLTK dependencies:

```
pip install nltk
```

## Functions

### 1. Text Preprocessing Functions

- **Lowercase Function:**
  ```
  def lowercase(input):
      return input.lower()
  ```

- **Remove Punctuation Function:**
  ```
  def remove_punctuation(input):
      return input.translate(str.maketrans('', '', string.punctuation))
  ```

- **Remove Whitespace Function:**
  ```
  def remove_whitespaces(input):
      return " ".join(input.split())
  ```

- **Remove HTML Tags Function:**
  ```
  def remove_html_tags(input):
      soup = BeautifulSoup(input, "html.parser")
      return soup.get_text(separator=" ")
  ```

- **Tokenization Function:**
  ```
  def tokenize(input):
      return word_tokenize(input)
  ```

- **Remove Stop Words Function:**
  ```
  def remove_stop_words(input):
      return [word for word in word_tokenize(input) if word not in stopwords.words('english')]
  ```

- **Lemmatization Function:**
  ```
  def lemmatize(input):
      lemmatizer = WordNetLemmatizer()
      return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(input)])
  ```

- **NLP Pipeline Function:**
  ```
  def nlp_pipeline(input):
      return lemmatize(' '.join(remove_stop_words(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(input)))))))
  ```

### 2. Topic Tagging Function

- **Topic Tagging with LDA Function:**
  ```
  def find_topics(question_body):
      try:
          text = nlp_pipeline(question_body)
          count_vectorizer = CountVectorizer(stop_words='english')
          count_data = count_vectorizer.fit_transform([text])
          
          # Configure LDA model
          number_topics = 1
          number_words = 2
          lda = LDA(n_components=number_topics, n_jobs=-1)
          lda.fit(count_data)
          
          # Extract topics
          words = count_vectorizer.get_feature_names()
          topics = [[words[i] for i in topic.argsort()[:-number_words - 1:-1]] for (topic_idx, topic) in enumerate(lda.components_)]
          topics = np.array(topics).ravel()
          
          # Filter topics that exist in the predefined set of tags
          existing_topics = set.intersection(set(topics), unique_tags)
  
      except Exception as e:
          print(f"Error processing question: {e}")
          return None
      
      return existing_topics
  ```

## Usage

1. **Import Functions:**

   Import the necessary functions into your Python script:

   ```
   from bs4 import BeautifulSoup
   import string
   from nltk.tokenize import word_tokenize
   from nltk.corpus import stopwords
   from nltk.stem import WordNetLemmatizer
   from dateutil import parser
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

2. **Function Calls:**

   Utilize the functions as needed in your application. For example, to find topics/tags associated with a question:

   ```
   question_body = "How can I learn data science effectively?"
   topics = find_topics(question_body)
   print("Topics:", topics)
   ```

   Adjust inputs and configurations based on your specific requirements and datasets.

## Acknowledgments

- This project utilizes NLTK and scikit-learn libraries for natural language processing tasks.
- The Latent Dirichlet Allocation (LDA) technique is used for topic modeling.


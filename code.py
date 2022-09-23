#Import functions for NLP
from bs4 import BeautifulSoup #to can deal with xml and html files 
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil import parser
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#Functions for NLP


def lowercase(input):
  """
  Returns lowercase text
  """
  return input.lower()

def remove_punctuation(input):
  """
  Returns text without punctuation
  """
  return input.translate(str.maketrans('','', string.punctuation))

def remove_whitespaces(input):
  """
  Returns text without extra whitespaces
  """
  return " ".join(input.split())
  
def remove_html_tags(input):
  """
  Returns text without HTML tags
  """
  soup = BeautifulSoup(input, "html.parser")
  stripped_input = soup.get_text(separator=" ")
  return stripped_input

def tokenize(input):
  """
  Returns tokenized version of text
  """
  return word_tokenize(input)

def remove_stop_words(input):
  """
  Returns text without stop words
  """
  input = word_tokenize(input)
  return [word for word in input if word not in stopwords.words('english')]

def lemmatize(input):
  """
  Lemmatizes input using NLTK's WordNetLemmatizer
  """
  lemmatizer=WordNetLemmatizer()
  input_str=word_tokenize(input)
  new_words = []
  for word in input_str:
    new_words.append(lemmatizer.lemmatize(word))
  return ' '.join(new_words)


def nlp_pipeline(input):
  """
  Function that calls all other functions together to perform NLP on a given text
  """
  return lemmatize(' '.join(remove_stop_words(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(input)))))))




################################
from sklearn.feature_extraction.text import CountVectorizer # to turn our text into a matrix of token counts 
from sklearn.decomposition import LatentDirichletAllocation as LDA

#Turn tags into a set for faster checking of whether a tag exists or not
unique_tags = set(tags['tags_tag_name'])

def find_topics(question_body):
  """
  Function that takes a question as an input, and finds the two most important topics/tags
  If the found topics exist in the already existing database of tags, we add these tags
  to the professional who answered the question
  """
  try:
    text = nlp_pipeline(question_body)
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform([text])
    # One topic that has an avg of two words because most questions had 1/2 tags
    number_topics = 1
    number_words = 2
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    words = count_vectorizer.get_feature_names()

    #Get topics from model. They are represented as a list e.g. ['military','army']
    topics = [[words[i] for i in topic.argsort()[:-number_words - 1:-1]] for (topic_idx, topic) in enumerate(lda.components_)]
    topics = np.array(topics).ravel()
    #Only use topics for which a tag already exists
    existing_topics = set.intersection(set(topics),unique_tags)
  
  #Three question bodies don't work with LDA so this exception just prints them out and ignores them
  except:
    print(question_body)
    return (question_body)

  return existing_topics

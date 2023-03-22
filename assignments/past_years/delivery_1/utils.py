from tqdm import tqdm
import re
from scipy.sparse import csr_matrix
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
import scipy

'''
DEFINITION OF THE DIFFERENT FUNCTIONS AND CLASSES USED IN TASK 1 NLP
'''

# Function to get list of strings
def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    #assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings

'''
CLASSES FOR FEATURE VECTORS (COUNT VECTORIZER AND TF-IDF)
'''

# Class for count vectorizer feature vector
class count_vectorizer:
    
    def __init__(self, sentences, tokken_pattern = r'(?u)\b\w\w+\b', lower_case = True, stop_words = False, stemming = False, lemmatization = False):
        
        
        self.lemmatization = lemmatization
        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        self.stemming = stemming
        if self.stemming:
            self.st = SnowballStemmer('english')
        
        
        if stop_words:
            self.stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "doing", "a", "an", "the", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "again", "further", "then", "once", "here", "there", "own", "same", "so", "than", "s", "t", "can", "will", "just", "don", "should", "now"]
        else:
            self.stop = []
        
        self.tokken_pattern = tokken_pattern
        self.lower = lower_case
        self.documents = sentences
        self.N = len(self.documents)
        #self.sentences = ' '.join(sentences)
        #self.all_words = re.findall(r'(?u)\b\w\w+\b', self.sentences) #Get all the words that satisfy the regular expression
        self.words = [] #List of all different words
        self.word2index = {} #Dictionary from word to his index
        
        
        return
    
    def fit(self):
        for document in tqdm(self.documents):
            if self.lower:
                document = str(document).lower()
            all_words = re.findall(self.tokken_pattern, document)
            for word in all_words:
                if self.stemming:
                    word = self.st.stem(word)
                if self.lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if word not in self.stop:
                    if word in self.word2index.keys():
                        pass
                    else:
                        self.words.append(word)
                        self.word2index[word] = len(self.words) - 1                  
        return
    
    def transform(self, sentences):
        row = []
        col = []
        data = []
        i = 0 #Defines number of document (row in sparse matrix)
        for document in tqdm(sentences):
            if self.lower:
                document = str(document).lower()
            all_words = re.findall(self.tokken_pattern, str(document))
            for word in set(all_words):
                if self.stemming:
                    word = self.st.stem(word)
                if self.lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if word not in self.stop:
                    if word in self.words:
                        row.append(i) #index representing number of the word
                        col.append(self.word2index[word]) #Column of the word (index)
                        data.append(all_words.count(word)) #Number of times word appears in sentence
            i += 1
        return csr_matrix((data, (row, col)), shape=(len(sentences), len(self.words)))

# Function for TF-IDF feature vector
class tf_idf:

    def __init__(self, sentences, tokken_pattern = r'(?u)\b\w\w+\b', lower_case = True, stop_words = False, stemming = False, lemmatization = False):
        
        
        self.lemmatization = lemmatization
        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        self.stemming = stemming
        if self.stemming:
            self.st = SnowballStemmer('english')
        
        if stop_words:
            self.stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "doing", "a", "an", "the", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "again", "further", "then", "once", "here", "there", "own", "same", "so", "than", "s", "t", "can", "will", "just", "don", "should", "now"]
        else:
            self.stop = []
        self.tokken_pattern = tokken_pattern
        self.lower = lower_case
        self.documents = sentences
        self.N = len(self.documents)
        #self.sentences = ' '.join(sentences)
        #self.all_words = re.findall(r'(?u)\b\w\w+\b', self.sentences) #Get all the words that satisfy the regular expression
        self.words = [] #List of all different words
        self.word2index = {} #Dictionary from word to his index
        self.count_word = []
        

        return


    def fit(self):
        for document in tqdm(self.documents):
            if self.lower:
                document = str(document).lower()
            all_words = re.findall(self.tokken_pattern, document)
            for word in all_words:
                if self.stemming:
                    word = self.st.stem(word)
                if self.lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if word not in self.stop:
                    if word in self.word2index.keys():
                        pass
                    else:
                        self.words.append(word)
                        self.word2index[word] = len(self.words) - 1
                        self.count_word.append(0)
            for word in set(all_words):
                if self.stemming:
                    word = self.st.stem(word)
                if self.lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if word not in self.stop:
                    self.count_word[self.word2index[word]] += 1
        self.idf = np.log(self.N / (1 + np.array(self.count_word)))
                    
        return
    
    def transform(self, sentences):
        row = []
        col = []
        data = []
        i = 0 #Defines number of document (row in sparse matrix)
        for document in tqdm(sentences):
            if self.lower:
                document = str(document).lower()
            all_words = re.findall(self.tokken_pattern, str(document))
            for word in set(all_words):
                if self.stemming:
                    word = self.st.stem(word)
                if self.lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if word not in self.stop:
                    if word in self.words:
                        row.append(i) #index representing number of the word
                        col.append(self.word2index[word]) #Column of the word (index)
                        data.append(np.log(all_words.count(word) + 1)) #Number of times word appears in sentence
            i += 1
        return csr_matrix((data, (row, col)), shape=(len(sentences), len(self.words))).multiply(csr_matrix(self.idf))

'''
FUNCTIONS FOR DIFFERENT STRATEGIES OF USING THE FEATURE VECTORES
'''

# Horitzontal stack of features of question 1 and 2
def stack_features(feat1, feat2):
    return scipy.sparse.hstack((feat1,feat2))

# Absolute difference of features of question 1 and 2
def difference(feat1, feat2):
    return abs(feat1 - feat2)

# Cosine similarity between both feature vectors
def similarity(feat1, feat2, dist = 'cosine'):
    if dist == 'cosine':
        dif = (feat1.multiply(feat2).sum(axis = 1).squeeze(-1))
        q1 = np.sqrt(feat1.multiply(feat1).sum(axis = 1)).squeeze(-1)
        q2 = np.sqrt(feat2.multiply(feat2).sum(axis = 1)).squeeze(-1)
        distances = np.multiply(dif, 1/(np.multiply(q1, q2) + 0.0001))
        return np.reshape(np.asarray(distances), (feat1.shape[0], 1))

# Difference between two feature vectors that returns a 0 if there's not the word for both questions,
# a -1 if only appears in one question and if it appears in both questions  the product -1 if the number of appearances is different
def different_product(feat1, feat2):
    return -(feat1 != feat2).astype(int) + feat1.multiply(feat2)


'''
FUNCTIONS TO CREATE EXTRA FEATURES FROM QUESTIONS
'''
def Variable_KeyWords(df):
    q1_list_train = cast_list_as_strings(list(df.question1))
    q2_list_train = cast_list_as_strings(list(df.question2))
    Key_words = ["how", "who", "where", "when", "why", "whom", "which", "whose", "what"]
    q1_feature = []
    q2_feature = []
    for k in range(len(q1_list_train)):
        if any(x in q1_list_train[k].lower().split(" ") for x in Key_words):
            q1_feature.append([word for word in q1_list_train[k].lower().split(" ") if word in Key_words])
        else:
            q1_feature.append("")
        
    for k in range(len(q2_list_train)):
        if any(x in q2_list_train[k].lower().split(" ") for x in Key_words):
            q2_feature.append([word for word in q2_list_train[k].lower().split(" ") if word in Key_words])
        else:
            q2_feature.append(" ")
        
    Variable_X = []
    for k in range(len(q1_feature)):
        if q1_feature[k] == q2_feature[k]:
            Variable_X.append(1)
        else:
            Variable_X.append(0) 
    
    return np.reshape(np.array(Variable_X), (np.array(Variable_X).shape[0], 1))


def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def get_jaccard(q1, q2):
    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    Float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(q1), set(q2)
    return 1 -  try_divide(len(set1 & set2), float(len(set1 | set2)))

def get_dice(q1,q2):
    
    q1, q2 = set(q1), set(q2)
    intersect = len(q1 & q2)
    union = float(len(q1) + len(q2))
    d = try_divide(2 * intersect, union)
    return d

def get_sorensen(q1, q2):
    """Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
    Float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(q1), set(q2)
    return 1-  try_divide(2 * len(set1 & set2),float(len(set1) + len(set2)))

def get_count_words_in_both(q1, q2):
    set1, set2 = set(q1), set(q2)
    return len(set1 & set2)

def get_ratio_words_in_both(q1, q2):
    set1, set2 = set(q1), set(q2)
    try:
        return len(set1 & set2)/float(len(set1))
    except:
        return 0.0

def get_num_of_words(q1):
    return len(q1)

def get_num_of_unique_words(q1):
    set1 = set(q1)
    return len(set1)

def get_count_of_digit(q1):
    return sum([1. for k in q1 if k.isdigit()])

def get_ratio_of_digit(q1):
    try:
        return sum([1. for k in q1 if k.isdigit()])/float(len(q1))
    except:
        return 0.0

def get_sim_feature(q1, q2):

    X_jaccard = np.array([ get_jaccard(x1, x2) for x1,x2 in zip(q1, q2)]).reshape(-1,1)
    X_dice = np.array([ get_dice(x1, x2)  for x1,x2 in zip(q1, q2)]).reshape(-1,1)
    X_count = np.array([ get_count_words_in_both(x1, x2)  for x1, x2 in zip(q1, q2)]).reshape(-1,1)
    X_ratio = np.array([ get_ratio_words_in_both(x1, x2)  for x1, x2 in zip(q1, q2)]).reshape(-1,1)
    X_len1 = np.array([ get_num_of_words(x1)  for x1 in  q1]).reshape(-1,1)
    X_len2 = np.array([ get_num_of_words(x2)  for x2 in  q2]).reshape(-1,1)

    X_len1_unique = np.array([ get_num_of_unique_words(x1)  for x1 in  q1]).reshape(-1,1)
    X_len2_unique = np.array([ get_num_of_unique_words(x2)  for x2 in  q2]).reshape(-1,1)

    X_len_diff = np.abs(X_len2-X_len1)


    X_sim = np.hstack([X_jaccard,X_dice,X_count,X_ratio,X_len1,X_len2,X_len1_unique,X_len2_unique,X_len_diff])
    

    return X_sim


'''
FUNCTIONS TO PRINT ERRORS
'''

def get_mistakes(clf, X_q1q2, y):

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions
    
def print_mistake_k(k, dataframe, mistake_indices, predictions):
    print(dataframe.iloc[mistake_indices[k]].question1)
    print(dataframe.iloc[mistake_indices[k]].question2)
    print("true class:", dataframe.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])
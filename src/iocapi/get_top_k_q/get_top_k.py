import operator
import os
import sys

import gensim
import _pickle as pickle
from get_top_k_q.algorithm import recommendation, similarity
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer


w2v = None
idf = None
questions = None
javadoc = None
javadoc_dict_classes = None
javadoc_dict_methods = None


def load_data():
    global w2v, idf, questions, javadoc, javadoc_dict_classes, javadoc_dict_methods
    current_dir = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(current_dir)

    w2v_path = os.path.join(current_dir, 'data', 'w2v_model_stemmed')
    idf_path = os.path.join(current_dir, 'data', 'idf')
    questions_path = os.path.join(current_dir, 'data', 'api_questions_pickle_new')
    javadoc_path = os.path.join(current_dir, 'data', 'javadoc_pickle_wordsegmented')

    if w2v is None:
        w2v = gensim.models.Word2Vec.load(w2v_path)  # pre-trained word embedding
    if idf is None:
        idf = pickle.load(open(idf_path, 'rb'))  # pre-trained idf value of all words in the w2v dictionary
    if questions is None:
        questions = pickle.load(open(questions_path, 'rb'))  # the pre-trained knowledge base of api-related questions (about 120K questions)
        questions = recommendation.preprocess_all_questions(questions, idf, w2v)  # matrix transformation
    if javadoc is None:
        javadoc = pickle.load(open(javadoc_path, 'rb'))  # the pre-trained knowledge base of javadoc
        javadoc_dict_classes = dict()
        javadoc_dict_methods = dict()
        recommendation.preprocess_javadoc(javadoc, javadoc_dict_classes, javadoc_dict_methods, idf, w2v)  # matrix transformation


def get_top_Q_A(query):
    load_data()
    query = query.replace('"', '').replace("'", '').replace('.', '').replace('?', '')
    query_words = WordPunctTokenizer().tokenize(query.lower())
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]
    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 1, dict())
    top_q_id = max(top_questions.items(), key=operator.itemgetter(1))[0]
    q = next((question for question in questions if question.id==top_q_id), None)
    print(q)
    return (q.title)

def get_top_k_apis(query, k):
    load_data()
    query = query.replace('"', '').replace("'", '').replace('.', '').replace('?', '')
    query_words = WordPunctTokenizer().tokenize(query.lower())
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]
    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, dict())
    return recommendation.recommend_api(query_matrix, query_idf_vector,
                                        top_questions, questions, javadoc, javadoc_dict_methods,k)

if __name__ == '__main__':
    apis = get_top_k_apis('How do I convert a String to an int in Java', 10)
    for api in apis:
        print(api)

    # [How do I read / convert an InputStream into a String in Java?](https://stackoverflow.com/questions/309424/how-do-i-read-convert-an-inputstream-into-a-string-in-java)
    # [How do I get the file name from a String containing the Absolute file path?](https://stackoverflow.com/questions/14526260/how-do-i-get-the-file-name-from-a-string-containing-the-absolute-file-path)
    # [How to combine paths in Java?](https://stackoverflow.com/questions/412380/how-to-combine-paths-in-java)
    # [How do I convert a String to an int in Java](https://stackoverflow.com/questions/5585779/how-do-i-convert-a-string-to-an-int-in-java)
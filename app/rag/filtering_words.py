import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng') 

lemmatizer = WordNetLemmatizer()

def filtering_incoming_words(query):
    tokens = word_tokenize(query)
    tagged_tokens = pos_tag(tokens)

    def get_wordnet_pos(tag):
        
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    lemmatized_sentence = []
    for word, tag in tagged_tokens:
        if word.lower() == 'are' or word.lower() in ['is', 'am']:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    result = ' '.join(lemmatized_sentence)

    return result

sentence = "The children are running towards a better place."

print(filtering_incoming_words(sentence))
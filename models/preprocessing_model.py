import pickle
from collections import Counter
import math
import re
import string 
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from num2words import num2words
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary, StopWordRemover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi komponen preprocessing
        self.stopword_factory = StopWordRemoverFactory()
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        # Setup stopwords
        stopword_data = self.stopword_factory.get_stop_words()
        more_stopword = ['jumlah', 'sebanyak', 'lainnya']
        stopword_data = stopword_data + more_stopword
        dictionary = ArrayDictionary(stopword_data)
        self.stopword = StopWordRemover(dictionary)
        
        # Setup angka teks
        self.angka_teks = set(num2words(i, lang='id') for i in range(0, 101))
        
        # Roman numeral pattern
        self.roman_pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        
    def case_folding(self, text):
        return text.lower()
    
    def tokenizing(self, text):
        return word_tokenize(text)
    
    def is_roman_numeral(self, word):
        return bool(re.match(self.roman_pattern, word.upper()))
    
    def roman_to_int(self, words):
        roman_values = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        ignore_words = ["di", "ke", "dari", "dan"]
        result = []

        for i, word in enumerate(words):
            prev_word = words[i - 1] if i > 0 else ""
            
            if word.lower() in ignore_words:
                result.append(word)
            elif self.is_roman_numeral(word):
                if prev_word.isdigit():
                    result.append(word)
                else:
                    total = 0
                    previous_value = 0
                    for char in reversed(word.upper()):
                        current_value = roman_values[char]
                        if current_value < previous_value:
                            total -= current_value
                        else:
                            total += current_value
                        previous_value = current_value
                    result.append(str(total))
            else:
                result.append(word)

        return result
    
    def number_to_text(self, word):
        return num2words(int(word), lang='id')
    
    def fungsi_terbilang(self, words):
        result = []
        
        for word in words:
            if "." in word and word.replace(".", "").isdigit():
                word = word.replace(".", "")
                result.append(self.number_to_text(word))
                continue
            
            if "." in word:
                word = word.replace(".", "")
                result.append(word)
                continue
            
            if "%" in word:
                word = word.replace("%", "persen")
                result.append(word)
                continue
            
            if word.startswith(",") or word.endswith(","):
                word = word.replace(",", "")
                result.append(word)
                continue
                
            if re.match(r'\d+,\d+$', word):
                before_comma, after_comma = word.split(',')
                before_comma = self.number_to_text(before_comma)
                after_comma = self.number_to_text(after_comma)
                result.append(before_comma + " koma " + after_comma)
                continue
            
            if re.match(r'^ke-\d+$', word):
                prefix, number = word.split('-')
                result.append(prefix + self.number_to_text(int(number)))
                continue
            
            if word.isdigit():
                word = self.number_to_text(int(word))
                result.append(word)
                continue
            
            result.append(word)
        return result
    
    def remove_stopwords(self, words):
        return [word for word in words if word in self.stopword.remove(word) or word in self.angka_teks]
    
    def stemming(self, words):
        stemmed_words = [self.stemmer.stem(word) for word in words]
        stemmed_words = ['capai' if word == 'capa' else word for word in stemmed_words]
        return stemmed_words
    
    def preprocess(self, text):
        # Case folding
        text = self.case_folding(text)
        
        # Tokenizing
        tokens = self.tokenizing(text)
        
        # Roman to int
        tokens = self.roman_to_int(tokens)
        
        # Number to text
        tokens = self.fungsi_terbilang(tokens)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stemming
        tokens = self.stemming(tokens)
        
        return ' '.join(tokens)

def save_preprocessor(preprocessor, filename):
    """Save the preprocessor object to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(preprocessor, f)

def load_preprocessor(filename):
    """Load the preprocessor object from a file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

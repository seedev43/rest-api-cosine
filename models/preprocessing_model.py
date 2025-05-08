import pickle
from collections import Counter
import math
import re
import string 
import nltk
import os

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)
# nltk.download('punkt_tab')
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
        more_stopword = [
            'jumlah', 'sebanyak', 'lainnya', 'lebih', 'melakukan',
            'kata', 'melalui', 'mengatakan', 'sebesar', 'terbesar',
            'digunakan', 'keterangan', 'tertulis', 'dipakai', 'pengeluaran',
            'dikarenakan', 'karena', 'hari', 'awal'
        ]
        stopword_data = stopword_data + more_stopword
        dictionary = ArrayDictionary(stopword_data)
        self.stopword = StopWordRemover(dictionary)
        
        
        # Setup angka teks
        self.angka_teks = set(num2words(i, lang='id') for i in range(0, 101))
        
        # Roman numeral pattern
        self.roman_pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
        
    def case_folding(self, word):
        return word.lower()
    
    def tokenizing(self, word):
        # punctuation = r'[.,;:!?()\[\]{}"\'/\\]'
  
        # # Memisahkan tanda baca dengan spasi
        # word = re.sub(f'({punctuation})', r' \1 ', word)
        
        # # Menghapus spasi berlebih
        # word = re.sub(r'\s+', ' ', word).strip()
        
        # Memisahkan kata-kata
        # return word.split()
        # print(word_tokenize(word))
        tokens = re.findall(r'\d{1,3}(?:\.\d{3})+|\d+,\d+|\w+|%|[^\w\s]', word, re.UNICODE)
        # return tokens
        return word_tokenize(word)

    def remove_punctuation(self, tokens):
        cleaned_tokens = [
            token for token in tokens
            if (
                re.match(r'^\d{1,3}(?:\.\d{3})+$', token)  # angka bertitik
                or re.match(r'^\d+,\d+$', token)           # angka berkoma
                or re.match(r'^\w+$', token)               # kata biasa
                or token == '%'                            # SIMBOL % DIPERTAHANKAN
            )
        ]
        # print(cleaned_tokens)
        return cleaned_tokens
    
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
        filtered_words = []
        for word in words:
            # If word is a number word, keep it
            if word in self.angka_teks:
                filtered_words.append(word)
            else:
                # Remove stopwords using Sastrawi
                processed_word = self.stopword.remove(word)
                if processed_word:  # If word is not a stopword
                    filtered_words.append(word)
        
        return filtered_words 
        return [word for word in words if word in self.stopword.remove(word) or word in self.angka_teks]
    
    def stemming(self, words):
        stemmed_words = [self.stemmer.stem(word) for word in words]
        replacements = {
            'capa': 'capai',
            'bas': 'belas',
        }
        
        # Mengganti kata-kata sesuai dengan dictionary
        stemmed_words = [replacements.get(word, word) for word in stemmed_words]
        # stemmed_words = ['capai' if word == 'capa' else word for word in stemmed_words]
        return stemmed_words
    
    def preprocess(self, text):
        # Case folding
        text = self.case_folding(text)
        
        # Tokenizing
        tokens = self.tokenizing(text)

        # Remove punctuation
        # tokens = self.remove_punctuation(tokens)
        
        # Roman to int
        tokens = self.roman_to_int(tokens)
        
        # Number to text
        tokens = self.fungsi_terbilang(tokens)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stemming
        tokens = self.stemming(tokens)
        
        return ' '.join(tokens)

# def save_preprocessor(preprocessor, filename):
#     """Save the preprocessor object to a file"""
#     with open(filename, 'wb') as f:
#         pickle.dump(preprocessor, f)

# def load_preprocessor(filename):
#     """Load the preprocessor object from a file"""
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

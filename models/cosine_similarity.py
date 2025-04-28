from collections import Counter
import math


def calculate_cosine_similarity(text1, text2):
    # Tokenisasi dan representasi vektor menggunakan Counter untuk menghitung frekuensi kata
    vector1 = Counter(text1.split())
    vector2 = Counter(text2.split())
    
    # print("Frekuensi kata pada kalimat 1:\n", vector1)
    # print("Frekuensi kata pada kalimat 2:\n", vector2)

    # Menggabungkan semua kata unik dalam kedua teks
    all_words = set(vector1.keys()).union(set(vector2.keys()))
    
    # print("Menggabungkan semua kata unik dalam kedua kalimat:\n", all_words)

    # Membuat vektor berdasarkan frekuensi kata
    vec1 = [vector1[word] for word in all_words]
    vec2 = [vector2[word] for word in all_words]
    
    # print(vec1, vec2)

    # Menghitung dot product antara dua vektor
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    
    # print(dot_product)

    # Menghitung norma vektor
    norm_vec1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
    norm_vec2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))

    # print(norm_vec1, norm_vec2)
    # Menghindari pembagian dengan nol
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    # Menghitung cosine similarity
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity

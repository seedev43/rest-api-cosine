from flask import Flask, request, jsonify, render_template
from models.preprocessing_model import TextPreprocessor
from models.cosine_similarity import calculate_cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

@app.route('/', methods=['GET'])
def main_page():
    return "Hello World"

@app.route('/form')
def form_page():
    return render_template('form.html')


@app.route('/similarity', methods=['POST'])
def check_similarity():
    preprocessor = TextPreprocessor()

    data = request.get_json()
    text1 = data.get('text1', "").strip()
    text2 = data.get('text2', "").strip()

    if not text1 or not text2:
        return jsonify({"success": False, "message": "Masukkan kalimat 1 dan kalimat 2 untuk memproses"}), 400

    
    process_text1 = preprocessor.preprocess(text1)
    process_text2 = preprocessor.preprocess(text2)

    # vectorizer = CountVectorizer()
    # converted_matrix = vectorizer.fit_transform([process_text1, process_text2])

    # Menghitung cosine similarity 
    # cosine_sim = cosine_similarity(converted_matrix[0], converted_matrix[1])[0][0]
    cosine_sim = calculate_cosine_similarity(process_text1, process_text2)

    threshold_similar = 0.8
    threshold_medium = 0.5
    status_similar = ""

    if cosine_sim >= threshold_similar:
        status_similar = "Kemiripan Tinggi"
    elif cosine_sim >= threshold_medium and cosine_sim < threshold_similar:
        status_similar = "Kemiripan Sedang"
    else:
        status_similar = "Tidak Mirip"

    return jsonify({
        "success": True,
        "result": {
            "original_text1": text1,
            "original_text2": text2,
            "processed_text1": process_text1,
            "processed_text2": process_text2,
            "similarity": cosine_sim,
            "similarity_percent": f"{cosine_sim * 100:.2f}%",
            "similarity_status": status_similar,
        },
    })


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run(debug=True) 

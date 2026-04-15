from flask import Flask, request, jsonify

app = Flask(__name__)

# Route 1: Home page
@app.route('/')
def home():
    return '<h1>Hello! Flask is working!</h1>'

# Route 2: Greet a user by namep
@app.route('/greet/<name>')
def greet(name):
    return f'<h2>Hello, {name}!</h2>'

# Route 3: Simple JSON API
@app.route('/api/data')
def get_data():
    data = {
        "status": "success",
        "message": "Flask API is running",
        "topics": [
            "Fake News Detection",
            "Named Entity Recognition",
            "Sentiment Analysis"
        ]
    }
    return jsonify(data)

# Route 4: Accept POST request
@app.route('/api/analyze', methods=['POST'])
def analyze():
    body = request.get_json()
    text = body.get('text', '')
    return jsonify({
        "received_text": text,
        "word_count": len(text.split()),
        "char_count": len(text)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
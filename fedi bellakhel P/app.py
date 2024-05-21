from flask import Flask, request, render_template
from flair.models import TextClassifier
from flair.data import Sentence

app = Flask(__name__)

# Load your Flair text classification model here
model = TextClassifier.load('offensive.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        text = request.form['text']
        sentence = Sentence(text)
        model.predict(sentence)
        prediction = sentence.labels[0]

        return render_template('result.html', text=text, prediction=prediction.value)

if __name__ == '__main__':
    app.run(debug=True)
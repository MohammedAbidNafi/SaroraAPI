from flask import Flask, request, jsonify

from Taylor.Taylor import chatWithTaylor

from Cheerful.Cheerful import chatWithCheerful

from Sarcastic.Sarcastic import chatWithSarcastic

app = Flask(__name__)


@app.route('/taylor', methods=['GET', 'POST'])
def ChatWithBot():
    chatInput = request.form['chatInput']
    return jsonify(Sarora=chatWithTaylor(chatInput))

@app.route('/cheerful', methods=['GET','POST'])
def ChatWithCheerful():
    chatInput = request.form['chatInput']
    return jsonify(Cheerful=chatWithCheerful(chatInput))

@app.route('/sarcastic',methods=['GET','POST'])
def ChatWithSarcastic():
    chatInput = request.form['chatInput']
    return jsonify(Cheerful=chatWithSarcastic(chatInput))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=13000, debug=True)

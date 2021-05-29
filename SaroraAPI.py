from flask import Flask, request, jsonify

from MainBot import chatWithBot

app = Flask(__name__)


@app.route('/sarora', methods=['GET', 'POST'])
def ChatWithBot():
    chatInput = request.form['chatInput']
    return jsonify(Sarora=chatWithBot(chatInput))


if __name__ == '__main__':
    app.run(host='192.168.1.39', debug=True)

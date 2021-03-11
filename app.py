import flask
from flask import jsonify
from analise import analise_texto

#texto_site = 'turno eleger partido politica'

app = flask.Flask(__name__, template_folder='templates')

@app.route("/", methods=['GET', 'POST'])
def main():
    resultado = 0
    html_wordcloud = ""
    html_freq = ""
    if flask.request.method == "POST":
        #print(flask.request.form['texto_analise'])
        resultado, html_wordcloud, html_freq = analise_texto(flask.request.form['texto_analise'])
    return flask.render_template('pages/dashboard.html',resultado = resultado, html_wordcloud = html_wordcloud, html_freq = html_freq)

@app.route("/sobre")
def sobre():
    return flask.render_template('pages/sobre.html')

@app.errorhandler(404)
def not_found(e):
    return flask.render_template('pages/404.html')

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
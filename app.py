from flask import Flask,render_template,url_for
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tumor')
def tumor():
    return render_template('first.html')

@app.route('/heart_attack')
def heart_attack():
    return render_template('second.html')

@app.route('/diabetes')
def diabetes():
    return render_template('third.html')

if __name__ =="__main__":
    app.run(debug=True)
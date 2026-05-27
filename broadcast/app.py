from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/music')
def music():
    return render_template('music.py')

if __name__ == '__main__':
    app.run(debug=True)

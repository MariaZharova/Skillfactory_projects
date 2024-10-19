from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def print_hello():
    print('hello, world!')

if __name__ == '__main__':
    app.run('127.5.5.10', 5000)

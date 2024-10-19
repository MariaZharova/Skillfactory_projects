from flask import Flask, request

app = Flask(__name__)

@app.route('/hello')
def hello_func():
    name_code = request.args.get('name_add', 'surname_add')
    #surname_code = request.args.get('surname_add')
    return f'Hello, {name_code}!'

@app.route('/')
def main_func():
    return 'Вы на главной странице!'

if __name__ == '__main__':
    app.run('localhost', 5000)

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    table = []
    number = None
    if request.method == 'POST':
        try:
            number = int(request.form['number'])
            table = [f"{number} x {i} = {number * i}" for i in range(1, 11)]
        except ValueError:
            table = ["Invalid input. Please enter a valid number."]
    return render_template('index.html', table=table, number=number)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
 
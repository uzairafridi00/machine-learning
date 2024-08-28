from flask import Flask, redirect, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello! this is the Main <h1>Index Page<h1>"

@app.route("/<name>")
def user(name):
    return f"Hello {name}"

@app.route("/dashboard")
def dashboard():
    return "<h1>Welcome to Dashboard<h1>"

@app.route("/admin")
def admin():
    return redirect(url_for("user", name="Admin!"))

if __name__ == "__main__":
    app.run()

from flask import Flask, render_template, request, redirect, url_for, session
from secret import secret_key
from datetime import timedelta

app = Flask(__name__)
app.secret_key = secret_key
app.permanent_session_lifetime = timedelta(minutes=20)
users = {'username':'password', 'username2':'password2', 'username3':'password3'}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        session.pop('user', None)
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        print(user)
        if user and user == password:
            session['user'] = username
            return redirect(url_for('profile'))
        return redirect(url_for('login'))
    if session.get('user'):
            username = session['user']
            print(username)
            return redirect(url_for('profile'))
    return render_template('login.html')

@app.route('/profile')
def profile():
    if session.get('user'):
        return render_template('profile.html', user=session['user'])
    return redirect(url_for('login'))

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
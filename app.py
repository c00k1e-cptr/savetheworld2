from flask import Flask, render_template, request, redirect, url_for, session
from secret import secret_key, salt
from datetime import timedelta
from database_handler import *
from hashlib import sha256

app = Flask(__name__)
app.secret_key = secret_key
app.permanent_session_lifetime = timedelta(minutes=20)

@app.route('/')
def index():
    if session.get('user'):
            username = session['user']
            return redirect(url_for('me'))
    return render_template('about.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        session.pop('user', None)
        username = request.form['username']
        password = request.form['password']
        users = get_all_users()
        user = users.get(username)
        if user and user ==  sha256((password+salt).encode('utf-8')).hexdigest():
            session['user'] = username
            return redirect(url_for('me'))
        return redirect(url_for('login'))
    if session.get('user'):
            username = session['user']
            return redirect(url_for('me'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = sha256((request.form['password']+salt).encode('utf-8')).hexdigest()
        confirm_password = sha256((request.form['confirm_password']+salt).encode('utf-8')).hexdigest()
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        if age:
            try:
                age = int(age)
            except:
                return render_template('signup.html', error='Age must be an number')
        height = request.form['height']
        if height:
            try:
                height = int(height)
            except:
                return render_template('signup.html', error='Height must be an number in centimeters')
        weight = request.form['weight']
        if weight:
            try:
                weight = int(weight)
            except:
                return render_template('signup.html', error='Weight must be an number in kilograms')
        gender = request.form['gender']
        if not (gender == '1' or gender == '0' or gender == '0.5'):
            return render_template('signup.html', error='Invalid gender')
        users = get_all_users()
        if users.get(username):
            return render_template('signup.html', error='Username already exists')
        session.pop('user', None)
        session['user'] = username
        insert_user(username, password, first_name, last_name, age, height, weight, gender)
        return redirect(url_for('login'))
    if session.get('user'):
            username = session['user']
            return redirect(url_for('me'))
    return render_template('signup.html', error='')

@app.route('/me')
def me():
    if session.get('user'):
        return render_template('profile.html', user=session['user'])
    return redirect(url_for('login'))

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
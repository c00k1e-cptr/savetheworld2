from flask import Flask, render_template, request, redirect, url_for, session
from secret import secret_key, salt
from datetime import timedelta
from database_handler import *
from hashlib import sha256
from model import predict

app = Flask(__name__)
app.secret_key = secret_key
app.permanent_session_lifetime = timedelta(minutes=30)


@app.route('/')
def index():
    if session.get('user'):
        username = session['user']
        return redirect(url_for('me'))
    return redirect(url_for('about'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user', None)
        username = request.form['username']
        if not username:
            return render_template('login.html', error='Username cannot be empty')
        password = request.form['password']
        if not password:
            return render_template('login.html', error='Password cannot be empty')
        users = get_all_users()
        user = users.get(username)
        if user:
            if user == sha256((password+salt).encode('utf-8')).hexdigest():
                session['user'] = username
                return redirect(url_for('me'))
            else:
                return render_template('login.html', error='Incorrect password')
        else:
            return render_template('login.html', error='User does not exist')
    if session.get('user'):
        username = session['user']
        return redirect(url_for('me'))
    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        if not username:
            return render_template('signup.html', error='Username cannot be empty')
        password = sha256(
            (request.form['password']+salt).encode('utf-8')).hexdigest()
        if not request.form['password']:
            return render_template('signup.html', error='Password cannot be empty')
        confirm_password = sha256(
            (request.form['confirm_password']+salt).encode('utf-8')).hexdigest()
        if not request.form['confirm_password']:
            return render_template('signup.html', error='Confirm Password cannot be empty')
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        first_name = request.form['first_name']
        if not first_name:
            return render_template('signup.html', error='First name cannot be empty')
        last_name = request.form['last_name']
        if not last_name:
            return render_template('signup.html', error='Last name cannot be empty')
        age = request.form['age']
        if age:
            try:
                age = int(age)
            except:
                return render_template('signup.html', error='Age must be an number')
        else:
            return render_template('signup.html', error='Age cannot be empty')
        height = request.form['height']
        if height:
            try:
                height = int(height)
            except:
                return render_template('signup.html', error='Height must be an number in centimeters')
        else:
            return render_template('signup.html', error='Height cannot be empty')
        weight = request.form['weight']
        if weight:
            try:
                weight = int(weight)
            except:
                return render_template('signup.html', error='Weight must be an number in kilograms')
        else:
            return render_template('signup.html', error='Weight cannot be empty')
        gender = request.form['gender']
        if not (gender == '1' or gender == '0' or gender == '0.5'):
            return render_template('signup.html', error='Invalid gender')
        users = get_all_users()
        if users.get(username):
            return render_template('signup.html', error='Username already exists')
        session.pop('user', None)
        session['user'] = username
        insert_user(username, password, first_name,
                    last_name, age, height, weight, gender)
        return redirect(url_for('login'))
    if session.get('user'):
        username = session['user']
        return redirect(url_for('me'))
    return render_template('signup.html', error=None)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        username = session['user']
        if not username:
            return render_template('profile.html', error='Error updating profile')
        user = get_user(username)
        password = sha256(
            (request.form['password']+salt).encode('utf-8')).hexdigest()
        if not request.form['password']:
            return render_template('profile.html', user=user, error='Password cannot be empty')
        confirm_password = sha256(
            (request.form['confirm_password']+salt).encode('utf-8')).hexdigest()
        if not request.form['confirm_password']:
            return render_template('profile.html', user=user, error='Confirm Password cannot be empty')
        if password != confirm_password:
            return render_template('profile.html', user=user, error='Passwords do not match')
        first_name = request.form['first_name']
        if not first_name:
            return render_template('profile.html', user=user, error='First name cannot be empty')
        last_name = request.form['last_name']
        if not last_name:
            return render_template('profile.html', user=user, error='Last name cannot be empty')
        age = request.form['age']
        if age:
            try:
                age = int(age)
            except:
                return render_template('profile.html', user=user, error='Age must be an number')
        else:
            return render_template('profile.html', user=user, error='Age cannot be empty')
        height = request.form['height']
        if height:
            try:
                height = int(height)
            except:
                return render_template('profile.html', user=user, error='Height must be an number in centimeters')
        else:
            return render_template('profile.html', user=user, error='Height cannot be empty')
        weight = request.form['weight']
        if weight:
            try:
                weight = int(weight)
            except:
                return render_template('profile.html', user=user, error='Weight must be an number in kilograms')
        else:
            return render_template('profile.html', user=user, error='Weight cannot be empty')
        gender = request.form['gender']
        if not (gender == '1' or gender == '0' or gender == '0.5' or gender == '0.0' or gender == '1.0'):
            return render_template('profile.html', user=user, error='Invalid gender')
        update_user(username, password, first_name,
                    last_name, age, height, weight, gender)
        user = get_user(username)
        return render_template('profile.html', user=user, error="Updated successfully")
    if session.get('user'):
        user = get_user(session['user'])
        return render_template('profile.html', user=user, error=None)
    return redirect(url_for('login'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/me')
def me():
    return render_template('dashboard.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    if session.get('user'):
        username = session['user']
        user = get_user(username)
        if request.method == 'GET':
            return render_template('test.html', error=None, user=user)
        else:
            age = request.form['age']
            if age:
                try:
                    age = int(age)
                except:
                    return render_template('test.html', user=user, error='Age must be an number')
            else:
                return render_template('test.html', user=user, error='Age cannot be empty')
            height = request.form['height']
            if height:
                try:
                    height = int(height)
                except:
                    return render_template('test.html', user=user, error='Height must be an number in centimeters')
            else:
                return render_template('test.html', user=user, error='Height cannot be empty')
            weight = request.form['weight']
            if weight:
                try:
                    weight = int(weight)
                except:
                    return render_template('test.html', user=user, error='Weight must be an number in kilograms')
            else:
                return render_template('test.html', user=user, error='Weight cannot be empty')
            hba1c = request.form['hba1c']
            if hba1c:
                try:
                    hba1c = float(hba1c)
                except:
                    return render_template('test.html', user=user, error='HbA1c (Hemoglobin A1c) level must be an decimal in kilograms')
            else:
                return render_template('test.html', user=user, error='HbA1c (Hemoglobin A1c) level cannot be empty')
            glucose = request.form['glucose']
            if glucose:
                try:
                    glucose = float(glucose)
                except:
                    return render_template('test.html', user=user, error='Blood glucose level must be an number in kilograms')
            else:
                return render_template('test.html', user=user, error='Blood glucose level cannot be empty')

            bmi = weight / ((height/100)**2)

            diabetes = int(predict([[age, bmi, hba1c, glucose]])[0])

            results = {
                0: 'Using the AI model, you likely do not have diabetes (YAY!). <br>However, to maintain that status here are some tips to stay healthy:<br><br>1. Balanced Diet: Eat a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit sugary foods and processed snacks.<br>2. Portion Control: Pay attention to portion sizes and listen to your hunger and fullness cues.<br>3. Regular Exercise: Aim for 150 minutes of moderate-intensity aerobic exercise per week, along with strength training twice a week.<br>4. Weight Management: Maintain a healthy weight through a balanced diet and regular physical activity.<br>5. Blood Pressure and Cholesterol: Monitor levels regularly and follow your doctor"s recommendations.<br>6. Stress Management: Practice stress-reducing techniques like meditation and deep breathing exercises.<br>7. Regular Check-ups: Schedule routine check-ups with your healthcare provider.', 
                1: 'Using the AI model, you likely have diabetes (OH NO!). <br>These are some ways to be healthy and try to reduce the symptoms<br><br>1. Balanced Diet: Follow a balanced diet consisting of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit sugary foods and carbohydrates.<br>2. Portion Control: Pay attention to portion sizes and space out your meals evenly throughout the day.<br>3. Carbohydrate Management: Monitor and manage your carbohydrate intake, considering the glycemic index of foods.<br>4. Regular Exercise: Engage in regular physical activity to improve insulin sensitivity and control blood sugar levels.<br>5. Medication and Insulin: Take prescribed medications or insulin as directed by your healthcare provider.<br>6. Blood Sugar Monitoring: Monitor your blood sugar levels regularly and keep a record of the readings.<br>7. Stress Management: Practice stress-reducing techniques such as meditation and relaxation exercises.<br>8. Regular Check-ups: Schedule regular check-ups with your healthcare provider and adhere to their recommendations for diabetes management.'}[diabetes]

            return render_template('test.html', user=user, error=None, results=results)

    return redirect(url_for('login'))


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

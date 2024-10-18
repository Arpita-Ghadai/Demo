from flask import Flask,render_template,request,redirect,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager,login_user,UserMixin,logout_user
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset for disease prediction
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data preparation and encoding for disease prediction
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Label encoding the prognosis (disease)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Train models for disease prediction
svc = SVC(probability=True)
nb = GaussianNB()
rf = RandomForestClassifier()

svc.fit(X_train, y_train)
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)

db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///my.db"
app.config["SECRET_KEY"]='thisissecret'
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String,nullable=False)
    password=db.Column(db.String,nullable=False)
    fname = db.Column(db.String,nullable=False)
    lname = db.Column(db.String,nullable=False)
    def __repr__(self):
        return '<User %r' % self.username

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        # Collecting input symptoms from the form
        symptoms = [int(request.form[f'symptom{i}']) for i in range(1, 6)]  # Getting the 5 symptoms
        symptoms = np.array([symptoms])

        # Predict using the three models
        svc_pred = svc.predict(symptoms)
        nb_pred = nb.predict(symptoms)
        rf_pred = rf.predict(symptoms)

        # Combine predictions (ensemble method)
        final_pred = np.bincount([svc_pred[0], nb_pred[0], rf_pred[0]]).argmax()

        disease = le.inverse_transform([final_pred])[0]
        return render_template('disease.html', prediction_text=f'Predicted Disease: {disease}')
    
    return render_template('disease.html')

@app.route('/clinic')
def clinic():
    return render_template('clinic.html')

@app.route('/register',methods=['POST','GET'])
def register():
    if request.method=='POST':
        email=request.form.get('email')
        password=request.form.get('password')
        username=request.form.get('username')
        fname=request.form.get('fname')
        lname=request.form.get('lname')
        user=User(username=username,email=email,password=password,fname=fname,lname=lname)
        db.session.add(user)
        db.session.commit()
        flash('User has been successfully registered','success')
        return redirect('/signin')

    return render_template('register.html')

@app.route('/signin',methods=['POST','GET'])
def signin():
    if request.method=='POST':
        username=request.form.get('username')
        password=request.form.get('password')
        user=User.query.filter_by(username=username).first()
        if user and password==user.password:
            login_user(user)
            return redirect('/home')
        else:
            flash('Invalid Credentials. Try again','danger')
            return redirect('/signin')
        
    return render_template('signin.html')

@app.route('/home',methods=['POST','GET'])
def home():
    return render_template('home.html')

if __name__=="__main__":
    with app.app_context():
      db.create_all()
    app.run(debug=True)
import os
from flask import Flask, redirect, url_for, request, render_template, session
from werkzeug.utils import secure_filename
import sqlite3
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras import backend as K

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for sessions, change this to a secure value

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load CNN model
model_path2 = 'xec.h5'
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

try:
    cnn_model = load_model(model_path2, custom_objects={'f1_score': f1_m, 'precision_score': precision_m, 'recall_score': recall_m}, compile=False)
    cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    print("CNN model loaded successfully")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    raise SystemExit("Cannot start app without CNN model.")

# Load trained classifiers
try:
    with open('hog_classifier.pkl', 'rb') as f:
        hog_classifier = pickle.load(f)
    with open('hybrid_classifier.pkl', 'rb') as f:
        hybrid_classifier = pickle.load(f)
    print("Classifiers loaded successfully")
except Exception as e:
    print(f"Error loading classifiers: {e}")
    raise SystemExit("Cannot start app without classifiers.")

# HOG Feature Extraction
def hog_feature_extraction(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.resize(image, (128, 128))
    image = cv2.equalizeHist(image)
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features.reshape(1, -1)

# Prediction Functions
def predict_cnn(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = np.argmax(cnn_model.predict(img))
        return "Forgery" if pred == 0 else "Genuine"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_hog(image_path):
    try:
        hog_features = hog_feature_extraction(image_path)
        pred = hog_classifier.predict(hog_features)
        return "Forgery" if pred == 0 else "Genuine"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_hybrid(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        cnn_features = cnn_feature_extractor.predict(img).flatten()
        hog_features = hog_feature_extraction(image_path).flatten()
        combined_features = np.concatenate([cnn_features, hog_features])
        pred = hybrid_classifier.predict(combined_features.reshape(1, -1))
        return "Forgery" if pred == 0 else "Genuine"
    except Exception as e:
        return f"Error: {str(e)}"

# Database Initialization
def init_db():
    try:
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS results 
                      (id INTEGER PRIMARY KEY, username TEXT, image_path TEXT, 
                       cnn_pred TEXT, hog_pred TEXT, hybrid_pred TEXT, timestamp TEXT)''')
        con.commit()
        con.close()
        print("Database initialized")
    except Exception as e:
        print(f"Database init error: {e}")

init_db()

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



@app.route('/index')
def index():
	return render_template('index.html')



@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "signatureotp75@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("signatureotp75@gmail.com", "dzag anni theh dtvn")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into info (user,email, password,mobile,name) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select user, password from info where user = ? AND password = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook1")
def notebook1():
    return render_template("CEDAR.html")

@app.route("/notebook2")
def notebook2():
    return render_template("UT_Sig.html")

@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file. Please upload a PNG, JPG, or JPEG image.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get predictions
   
        hybrid_pred = predict_hybrid(file_path)

    

        return render_template("result.html", 
                             hybrid_pred=hybrid_pred, 
                             img_src=file_path)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
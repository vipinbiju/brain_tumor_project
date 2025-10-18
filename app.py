import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
# IMPORTANT: In a real application, use an environment variable for the secret key
app.secret_key = 'a_very_secure_secret_key_for_brain_app' 

# Add context processor to make 'datetime' object available globally in all Jinja templates
@app.context_processor
def inject_global_vars():
    """Injects the Python datetime object into all template contexts."""
    return dict(datetime=datetime)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file upload security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_tumor_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# --- Model Loading and Configuration ---
# NOTE: The model path 'brain_tumor_model.keras' must exist in the execution environment.
try:
    model = load_model('brain_tumor_model.keras')
    logging.info("Deep learning model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # In a production environment, you might halt execution or use a dummy model.
    model = None 

# Define class names based on user request (Order MUST match model output order)
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    # NOTE: Passwords should be hashed in a real application
    password = db.Column(db.String(100), nullable=False) 

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), nullable=False)
    image_filename = db.Column(db.String(100), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False) # e.g., "Glioma Tumor (98.50%)"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created/checked.")

# ---------------- Utility Functions ----------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please try logging in.', 'danger')
            return redirect(url_for('register'))

        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['email'] = user.email
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('Logged out successfully. See you next time!', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'email' not in session:
        flash('Please login to access the prediction service.', 'warning')
        return redirect(url_for('login'))
        
    if model is None:
        flash('Prediction service is unavailable. Model failed to load.', 'danger')
        return redirect(url_for('dashboard')) # Redirect to a safe page

    if request.method == 'POST':
        file = request.files.get('image')
        filepath = None # Initialize filepath

        if not file or file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
            return redirect(request.url)

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get the expected input shape from the model (e.g., (224, 224))
            input_shape = model.input_shape[1:3] 
            
            # Load and preprocess image
            img = image.load_img(filepath, target_size=input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = img_array / 255.0 # Normalize pixel values

            # Predict
            prediction_array = model.predict(img_array)
            
            # Find the class with the highest probability
            predicted_class_index = np.argmax(prediction_array[0])
            result = CLASS_NAMES[predicted_class_index]
            
            # Get confidence for display
            confidence = float(prediction_array[0][predicted_class_index] * 100)
            
            # Store result with confidence for the dashboard
            full_result_text = f"{result} ({confidence:.2f}%)"
            
            # Save prediction
            new_prediction = Prediction(
                user_email=session['email'],
                image_filename=filename,
                prediction_result=full_result_text
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            # Use url_for for the image path for robustness
            img_url = url_for('static', filename=f'uploads/{filename}')

            return render_template('result.html', 
                                   result=result, 
                                   confidence=confidence,
                                   img_path=img_url)

        except Exception as e:
            logging.error(f'Prediction processing failed: {e}', exc_info=True)
            flash(f'An internal error occurred during prediction. Please check the uploaded file. Error: {e}', 'danger')
            # Clean up the file if an error occurred during processing
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return redirect(request.url)

    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        flash('Please login to view your dashboard.', 'warning')
        return redirect(url_for('login'))

    user_email = session['email']
    # Fetch user's history, ordered by newest first
    predictions = Prediction.query.filter_by(user_email=user_email).order_by(Prediction.timestamp.desc()).all()
    
    # Simple logic to determine color for status badges
    def get_status_color(result_text):
        if 'No Tumor' in result_text:
            return 'bg-green-100 text-green-800 ring-green-500'
        return 'bg-red-100 text-red-800 ring-red-500'

    return render_template('dashboard.html', 
                           predictions=predictions, 
                           get_status_color=get_status_color)

# ---------------- Run App ----------------
# The following block is typically used when running locally.
# In a deployment environment (like an external host or sandbox),
# the host typically handles the app startup (e.g., using gunicorn).
if __name__ == '__main__':
    app.run(debug=True)

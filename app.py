from flask import Flask, render_template, request, jsonify, redirect, url_for, session,Response
import json
import os
import bcrypt
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import google.generativeai as genai
from close_end_questionaire import get_random_close_questions
import csv
from open_end_questions import get_random_open_questions
import pandas as pd
from csv_extracter import csv_to_string
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from gemini_ai import gemini_chat
from image_analysis import analyze_image
from werkzeug.utils import secure_filename
from trained_wecare import chatbot_response
import cv2
import numpy as np
from csv_extracter import close_ended_response, open_ended_response
from markdown import markdown
import markdown2
from flask_mail import Mail, Message
import os
from fpdf import FPDF
from flask import Flask, request, jsonify
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from werkzeug.utils import secure_filename
from flask import send_file
from flask import request, jsonify, send_file
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import random
import time
from flask import flash  
from flask import Flask, render_template, session, redirect, url_for

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Use environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

api_keys = np.array([
    "AIzaSyBVRyfNmHvJBcqN6lHXY_H9GkXrlxMTO14",
    "AIzaSyDY8DmGDAT4_7FXVFyk8j5SEfFOSge-i38",
    "AIzaSyAtTbuoEYH_DkvYT4jGuLXlTQbknn2C9E4",
    "AIzaSyDnXKVCixgCwRwzq9OJ8FT5YD418Ce2puM"
])

DEFAULT_ANALYSIS = {
    "parameters": ["Anxiety", "Depression", "Self-esteem", "Stress", "Sleep Quality"],
    "scores": [50, 50, 50, 50, 50],
    "analysis": {
        "General Assessment": {
            "score": 50,
            "rationale": "Analysis unavailable due to temporary service issue. Please try again later."
        }
    },
    "suggestions": {
        "Immediate Actions": [
            "Practice deep breathing exercises",
            "Take a short walk",
            "Drink water"
        ],
        "Long-term Strategies": [
            "Maintain a regular sleep schedule",
            "Consider talking to a professional",
            "Practice mindfulness daily"
        ]
    },
    "well_being_tips": [
        "Take regular screen breaks",
        "Connect with friends/family",
        "Engage in light physical activity",
        "Maintain a balanced diet",
        "Practice gratitude journaling"
    ]
}

# Load environment variables
load_dotenv()
# Update mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587  # Changed to 587 for TLS
app.config['MAIL_USE_TLS'] = True  # Changed from SSL to TLS
app.config['MAIL_USERNAME'] = 'wecarefinalyearproject@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'wecarefinalyearproject@gmail.com'

mail = Mail(app)

def allowed_file(filename):
    """
    Checks if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Configure SQLAlchemy
db = SQLAlchemy(app)

# Initialize Gemini AI
genai.configure(api_key=os.getenv("API_KEY"))

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Create database tables
with app.app_context():
    db.create_all()


@app.context_processor
def inject_logo():
    return {'logo_name': 'WeCare'}

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        # Set the background image path
        self.background_png = "static/pdf_background.png"
        
        # Ensure the background image exists
        if not os.path.exists(self.background_png):
            raise FileNotFoundError(f"Background image not found at: {self.background_png}")
        
        # Configure default settings
        self.set_auto_page_break(True, margin=40)
        self.set_margins(20, 40, 20)  # Left, Top, Right margins

    def header(self):
        """Add the PNG background to all pages"""
        # Add background image (full page)
        self.image(
            self.background_png,
            x=0, y=0,
            w=210,  # A4 width in mm
            h=297   # A4 height in mm
        )
        
        # Add header content (titles, user info, etc.)
        self.set_y(40)  # Start content below header
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(74, 144, 226)  # Blue for title
        self.cell(0, 10, 'Mental Health Analysis Report', align='C', ln=1)
        self.ln(5)
        
        # User info
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)  # Black for text
        self.cell(0, 10, f'User: {self.user_name}', ln=1)
        self.cell(0, 10, f'Email: {self.user_email}', ln=1)
        self.ln(10)

    def add_disclaimer(self):
        """Add disclaimer section"""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(200, 0, 0)  # Red for disclaimer
        self.cell(0, 10, 'Disclaimer:', ln=1)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)  # Black for text
        self.multi_cell(0, 6, 'This report is for informational purposes only and does not constitute medical advice. Please consult a professional for any concerns.')
        self.ln(10)

    def add_radar_chart(self, chart_path):
        """Add radar chart with title"""
        if chart_path and os.path.exists(chart_path):
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(74, 144, 226)  # Blue for title
            self.cell(0, 10, 'Radar Chart:', ln=1)
            self.ln(5)
            
            # Add chart image
            self.image(chart_path, x=30, y=self.get_y(), w=150, h=150)
            self.ln(160)  # Space after chart

    def add_section(self, title):
        """Add section header"""
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(74, 144, 226)  # Blue for title
        self.cell(0, 10, title, ln=1)
        self.ln(5)

    def add_analysis_content(self, data):
        """Add analysis content with bullet points"""
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)  # Black for text
        
        for key, value in data.items():
            # Section header
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, f'{key}:', ln=1)
            
            # Content
            self.set_font('Helvetica', '', 12)
            if isinstance(value, dict):
                text = f"• {value.get('rationale', 'No rationale provided.')}"
            else:
                text = f"• {value}"
            
            self.multi_cell(0, 6, text)
            self.ln(4)  # Space between items

    def add_wellbeing_tips(self, tips):
        """Add well-being tips with bullet points"""
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)  # Black for text
        self.cell(0, 10, 'Well-Being Tips:', ln=1)
        self.ln(5)
        
        for tip in tips:
            self.multi_cell(0, 6, f"• {tip}")
            self.ln(4)  # Space between tips

    def footer(self):
        """Add footer with page number"""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)  # Gray for footer
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
def generate_radar_chart(analysis):
    """Generate radar chart matching client-side styling"""
    if not analysis or "parameters" not in analysis or "scores" not in analysis:
        return None

    params = analysis["parameters"]
    scores = analysis["scores"]
    
    try:
        # Complete the radar loop
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        angles += angles[:1]
        scores += scores[:1]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Plot styling
        ax.plot(angles, scores, color='#4a90e2', linewidth=2, linestyle='solid')
        ax.fill(angles, scores, color='#4a90e2', alpha=0.25)
        
        # Axis styling
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 100)
        
        # Label styling
        plt.xticks(angles[:-1], params, fontsize=10, color='#333')
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
                   color="gray", size=8)
        plt.grid(color='#eee', linestyle='--', linewidth=0.5)
        
        # Save chart
        chart_path = "static/temp_radar.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=150)
        plt.close()  # Explicitly close the figure
        return chart_path
        
    except Exception as e:
        print(f"Error generating radar chart: {str(e)}")
        return None
    finally:
        # Ensure figure is always closed
        plt.close('all')

def generate_pdf(user_name, user_email, analysis):
    pdf = PDFReport()
    pdf.user_name = user_name
    pdf.user_email = user_email
    
    # Add content to the PDF
    pdf.add_page()
    pdf.add_disclaimer()
    
    # Add radar chart
    chart_path = generate_radar_chart(analysis)
    pdf.add_radar_chart(chart_path)
    
    # Add analysis sections
    if "analysis" in analysis:
        pdf.add_section("Mental Health Analysis")
        pdf.add_analysis_content(analysis["analysis"])
    
    # Add suggestions
    if "suggestions" in analysis:
        pdf.add_section("Steps to Overcome Issues")
        pdf.add_analysis_content(analysis["suggestions"])
    
    # Add well-being tips
    if "well_being_tips" in analysis:
        pdf.add_section("Well-Being Tips")
        pdf.add_wellbeing_tips(analysis["well_being_tips"])
    
    # Save PDF
    output_path = "static/reports/Mental_Health_Report.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    
    return output_path

def generate_radar_chart(analysis):
    # Generate a radar chart image using matplotlib (if data provided)
    if analysis and "parameters" in analysis and "scores" in analysis:
        params = analysis["parameters"]
        scores = analysis["scores"]
        # Complete the loop for radar chart
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        angles += angles[:1]
        scores += scores[:1]
        
        fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
        ax.plot(angles, scores, color='#4a90e2', linewidth=2)
        ax.fill(angles, scores, color='#4a90e2', alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 100)
        plt.xticks(angles[:-1], params, fontsize=10)
        chart_path = "static/temp_radar.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        return chart_path
    return None

def generate_pdf(user_name, user_email, analysis):
    """
    Generates a PDF report that exactly mimics the provided jsPDF layout.
    
    Expected analysis dict should contain:
      - "analysis": dict of detailed analysis,
      - "suggestions": dict of steps to overcome issues,
      - "well_being_tips": list of tips,
      - Optionally, "parameters" and "scores" for the radar chart.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.header_bg()
    pdf.add_branding_and_user_info(user_name, user_email)
    pdf.add_disclaimer()
    
    # Radar Chart (if exists)
    chart_path = generate_radar_chart(analysis)
    if chart_path and os.path.exists(chart_path):
        pdf.check_page_break(150)
        current_y = pdf.get_y()
        # Position chart at x=35, width=140, height=140
        pdf.image(chart_path, x=35, y=current_y, w=140, h=140)
        pdf.ln(150)
    
    # Section: Mental Health Analysis
    if "analysis" in analysis:
        pdf.add_section("Mental Health Analysis")
        pdf.add_content(analysis["analysis"])
    
    # Section: Steps to Overcome Issues
    if "suggestions" in analysis:
        pdf.add_section("Steps to Overcome Issues")
        pdf.add_content(analysis["suggestions"])
    
    # Section: Well-Being Tips
    if "well_being_tips" in analysis:
        pdf.add_section("Well-Being Tips")
        for tip in analysis["well_being_tips"]:
            pdf.multi_cell(0, pdf.line_spacing, f"- {tip}")
            pdf.ln(4)
            pdf.check_page_break(10)
    
    # Save the PDF report
    output_path = "static/reports/Mental_Health_Report.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path

# Home Route
@app.route('/')
def index():
    return render_template('index.html')


# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='⚠️ Email already exists. Please log in.')

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        send_welcome_email(email, name)

        return redirect('/login')

    return render_template('register.html')



def send_welcome_email(email, name):
    try:
        msg = Message("Welcome to WeCare!", recipients=[email])
        msg.body = f"""
        Hi {name}, 
        
        Welcome to WeCare! We're delighted to have you join us. 
        
        Our mission is to support your well-being and mental health. 
        Feel free to explore our platform and take advantage of the resources available.

        If you ever need assistance, don't hesitate to reach out.

        Best regards,  
        The WeCare Team
        """

        # Attach the WeCare logo
        logo_path = os.path.join("static", "WeCare.jpg")
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as logo:
                msg.attach("WeCare.jpg", "image/jpeg", logo.read())

        # Send email
        mail.send(msg)
        print(f"✅ Welcome email sent to {email}!")

    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")



# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/session')

        else:
            return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')

@app.route('/closed_ended')
def closed_ended():
    email = session.get('email')
    old_responses = None
    if email:
        user_folder = get_user_folder(email)
        file_path = os.path.join(user_folder, "close_end_questions_responses.csv")
        if os.path.exists(file_path):
            # Optionally, you can load the responses and pass them to the template
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                old_responses = list(reader)
    random_questions = get_random_close_questions()  # Get random questions as usual
    return render_template('closed_ended.html', questions=random_questions, old_responses=old_responses)


@app.route('/submit_close_end', methods=['POST'])
def submit_close_ended():
    if request.method == 'POST':
        responses = []
        for question in request.form:
            answer = request.form[question]
            responses.append((question, answer))
        
        # Get the user's email from session
        email = session.get('email')
        if email:
            user_folder = get_user_folder(email)
            # Overwrite the file each time
            file_path = os.path.join(user_folder, "close_end_questions_responses.csv")
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Question', 'Answer'])  # header
                for response in responses:
                    writer.writerow(response)
        else:
            # Handle case if session has no email
            print("No email in session!")
        
        return redirect(url_for('submit_opended'))


def save_to_csv(responses):
    file_exists = os.path.isfile('responses/close_end_questions_responses.csv')

    with open('responses/close_end_questions_responses.csv', mode='w', newline='') as file:  # Changed to 'a'
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Question', 'Answer'])  # Write header if the file doesn't exist
        
        for response in responses:
            writer.writerow(response)

@app.route('/open_ended', methods=['GET', 'POST'])
def submit_opended():
    if request.method == 'POST':
        responses = {key: value for key, value in request.form.items()}
        print("Received responses:", responses)  # Debug: print received responses
        save_responses_to_csv(responses)
        return redirect(url_for('thank_you'))  # Redirect to the user's dashboard
    else:
        random_questions = get_random_open_questions()  # Load new questions if needed
        return render_template('open_ended.html', questions=random_questions)

def save_responses_to_csv(responses):
    """Saves open-ended responses to a CSV file in the user's dedicated folder."""
    # Get the user's email from the session
    email = session.get("email")
    if email:
        user_folder = get_user_folder(email)
        file_path = os.path.join(user_folder, "open_end_questions_responses.csv")
    else:
        # Fallback in case no email is available (shouldn't normally occur)
        file_path = os.path.join("responses", "open_end_questions_responses.csv")
    
    # Debug: print the file path being used
    print(f"Saving open-ended responses to: {file_path}")
    
    # Prepare the data as a list of tuples: (question, response)
    data_to_save = [(question, responses[question]) for question in responses]
    print("Data to save:", data_to_save)  # Debug: print the data to be saved
    
    # Create a DataFrame from the data and overwrite the file
    try:
        df = pd.DataFrame(data_to_save, columns=['Question', 'Response'])
        df.to_csv(file_path, mode='w', header=True, index=False)
        print("File saved successfully.")
    except Exception as e:
        print("Error saving CSV:", str(e))

@app.route('/session')
def session_page():
    email = session.get('email')

    if email:
        user = User.query.filter_by(email=email).first()
        user_name = user.name if user else "User"
    else:
        user_name = "Guest"

    session['user_name'] = user_name
    return render_template('session.html', user_name=user_name)

@app.route('/profile')
def profile():
    email = session.get('email')
    if email:
        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('profile.html', user=user)  # Render profile.html with user data
        else:
            return redirect(url_for('session_page')) # User not found, redirect to session
    else:
        return redirect(url_for('login'))  # Redirect to login if no email in session
    

@app.route('/thank_you')
def thank_you():
    # Check authentication
    email = session.get('email')
    if not email:
        flash("Please log in to view your feedback", "danger")
        return redirect(url_for('login'))

    # Get user details
    user = User.query.filter_by(email=email).first()
    if not user:
        flash("User not found", "danger")
        return redirect(url_for('login'))
    user_name = user.name

    # Get user-specific file paths
    user_folder = get_user_folder(email)
    close_file = os.path.join(user_folder, "close_end_questions_responses.csv")
    open_file = os.path.join(user_folder, "open_end_questions_responses.csv")

    # Validate response files
    if not all(os.path.exists(f) for f in [close_file, open_file]):
        flash("Complete both assessments to view feedback", "warning")
        return redirect(url_for('session_page'))

    # Read CSV data with error handling
    try:
        close_ended_str = csv_to_string(close_file)
        open_ended_str = csv_to_string(open_file)
        
        if "No data" in close_ended_str or "No data" in open_ended_str:
            raise ValueError("Empty response files")
            
    except Exception as e:
        print(f"Error reading responses: {str(e)}")
        flash("Error loading your responses", "danger")
        return redirect(url_for('session_page'))

    # Gemini feedback generation
    def fetch_gemini_feedback(prompt, keys, current_index=0):
        if current_index >= len(keys):
            raise Exception("All API keys exhausted")
        try:
            genai.configure(api_key=keys[current_index])
            return gemini_chat(prompt)
        except Exception as e:
            if 'quota' in str(e).lower():
                return fetch_gemini_feedback(prompt, keys, current_index + 1)
            raise

    try:
        # Structured prompt template
        main_prompt = f"""
        **Mental Health Assessment Analysis**
        Analyze this user's responses:

        **Closed-ended Answers:**
        {close_ended_str}

        **Open-ended Responses:**
        {open_ended_str}

        Provide feedback following these rules:
        1. Use compassionate, non-judgmental tone
        2. Structure in 3 sections:
           - Key Strengths (3 bullet points)
           - Areas for Growth (3 bullet points)
           - Actionable Recommendations (3 steps)
        3. Include relevant emojis for warmth
        4. Use markdown formatting with headers
        5. Keep under 250 words
        6. Add a positive closing statement
        """

        # Get feedback from Gemini
        judge_gemini = gemini_chat(main_prompt)
        summarize = fetch_gemini_feedback(main_prompt, api_keys)
        
        # Convert markdown to HTML
        summarize_html = markdown(summarize)
        
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        summarize_html = markdown("""
        ## 🛠 Feedback Generation Issue
        We're experiencing high demand. Please try:
        - Refreshing the page
        - Checking back in 5 minutes
        - Contacting support if issue persists
        """)

    return render_template(
        'thank_you.html',
        judge_gemini=summarize_html,
        user_name=user_name,
        completejudege=judge_gemini
    )

# Helper function for CSV conversion
def csv_to_string(file_path):
    """Converts CSV to formatted string with error handling"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return "No data in response file"
        return df.to_string(index=False)
    except Exception as e:
        print(f"CSV read error: {str(e)}")
        return "Could not read response data"

from gemini_ai import gemini_chat
from gemini_ai import model

@app.route('/logout')
def logout():
    session.pop('email', None)  # Remove email from session
    session.pop('user_name', None) # Remove username from session
    return redirect(url_for('login'))  # Redirect to login page


chat_session = model.start_chat()

# Chat function to handle user input
def gemini_chat(user_input, history_file="dataset/intents.json"):
    try:
        # Load intents from JSON or initialize
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                intents_data = json.load(f)
        else:
            intents_data = {"intents": []}

        # Send user input to the model
        response = chat_session.send_message(user_input)

        # Create a new intent object for the conversation
        new_intent = {
            "patterns": [user_input],
            "responses": [response.text.strip()],
        }

        # Append the new intent to the intents list
        intents_data['intents'].insert(1, new_intent)

        # Save the updated intents JSON file
        with open(history_file, 'w') as f:
            json.dump(intents_data, f, indent=4)

        return response.text

    except Exception as e:
       
        response = chatbot_response(user_input)
        # Optionally log the error to a file
        with open('error.log', 'a') as log_file:
            log_file.write(f"{str(e)}\n")
        return response
# in excecption the pretained model so if error occurs then it can use the pretrained model 
# Chat Route
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('main.html')
    elif request.method == 'POST':
        user_input = request.form['user_input']
        response = gemini_chat(user_input)  # Call gemini_chat function here
        return jsonify({'response': response})
    else:
        return "Unsupported request method", 405  # Handle other methods if needed


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.
you have to analayse the image but and predict the cause or whats that 
Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
3. Your insights are invaluable in guiding clinical decisions. Please proceed with analysis, adhering to the structured approach outlined above.
"""


@app.route('/image_analysis', methods=['GET', 'POST'])
def image_analysis():
    analysis = None
    image_path = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return redirect(request.url)

        # Validate file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        if '.' not in uploaded_file.filename or \
           uploaded_file.filename.split('.')[-1].lower() not in allowed_extensions:
            flash('Only PNG, JPG/JPEG, and WebP files are allowed', 'error')
            return redirect(request.url)

        if uploaded_file:
            # Create uploads directory if it doesn't exist
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Generate unique filename
            filename = secure_filename(f"{int(time.time())}_{uploaded_file.filename}")
            image_path = os.path.join(upload_dir, filename)
            uploaded_file.save(image_path)
            
            # Process the image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                
            image_parts = [{
                "mime_type": uploaded_file.mimetype,
                "data": image_data
            }]

            prompt_parts = [image_parts[0], system_prompt]
            response = model.generate_content(prompt_parts)

            # Convert response to Markdown format
            analysis = markdown2.markdown(response.text)

    return render_template('image_analysis.html', 
                         analysis=analysis, 
                         image_path=image_path.replace('\\', '/') if image_path else None)



import cv2
import numpy as np

# Initialize face detector using OpenCV's Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize global variables
attention_status = "Not Paying Attention"
dominant_emotion = "neutral"  # Default value

def is_paying_attention(face_detected):
    """ Checks if the user is paying attention based on face detection. """
    return face_detected

def detect_face_and_attention(frame):
    """ Detects face and attention from the frame. """
    global attention_status, dominant_emotion

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Get the first face
        (x, y, w, h) = faces[0]
        
        # Update attention status
        attention_status = "Paying Attention"
        
        # Draw bounding box and display attention status
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Face Detected ({attention_status})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        attention_status = "Not Paying Attention"

    return frame

# Video feed generator
def generate_frames():
    """ Captures frames from the webcam and detects face and attention. """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_face_and_attention(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


# Flask route for the chatbot interaction
@app.route('/talk_to_me', methods=['GET', 'POST'])
def talk_to_me():
    """ Handles the user's input and sends it to the chatbot along with emotion and attention. """
    global attention_status, dominant_emotion

    if request.method == 'GET':
        return render_template('talk_to_me.html')

    elif request.method == 'POST':
        user_input = request.form.get('user_input', '')

        # Create the prompt with the attention status and emotion
        prompt = f"The user is in a {dominant_emotion} mood and is {'paying attention' if attention_status == 'Paying Attention' else 'not paying attention'}."

        # Call the gemini_chat function with the user input and the generated prompt
        bot_response = gemini_chat(user_input + " " + prompt)

        return jsonify({'response': bot_response})

    else:
        return "Unsupported request method", 405

def log_conversation(user_input, bot_response, history_file="dataset/intents.json"):
    # Load or initialize intents data
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            intents_data = json.load(f)
    else:
        intents_data = {"intents": []}

    # Create a new intent object
    new_intent = {
        "patterns": [user_input],
        "responses": [bot_response],
    }

    # Append the new intent
    intents_data['intents'].append(new_intent)

    # Save the updated intents JSON file
    with open(history_file, 'w') as f:
        json.dump(intents_data, f, indent=4)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))




def send_to_gemini(close_path, open_path):
    """Analyzes user-specific CSV files with Gemini"""
    try:
        # Validate file existence
        if not all(os.path.exists(p) for p in [close_path, open_path]):
            print(f"⚠️ Missing files: {close_path} | {open_path}")
            return DEFAULT_ANALYSIS

        # Load data with error handling
        try:
            close_end_df = pd.read_csv(close_path)
            open_end_df = pd.read_csv(open_path)
        except pd.errors.EmptyDataError:
            print("❌ Empty CSV files detected")
            return DEFAULT_ANALYSIS

        # Validate data presence
        if close_end_df.empty or open_end_df.empty:
            print("⚠️ Empty response files")
            return DEFAULT_ANALYSIS

        # Convert to text with headers
        close_end_text = close_end_df.to_string(index=False, header=True)
        open_end_text = open_end_df.to_string(index=False, header=True)

        # Configure model with JSON response type
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        # Structured prompt template
        prompt = f"""Analyze these mental health responses:

        Close-ended Questions:
        {close_end_text}

        Open-ended Responses:
        {open_end_text}

        Generate JSON with this exact structure:
        {{
            "parameters": ["Anxiety", "Depression", "Self-esteem", "Stress", "Sleep Quality"],
            "scores": [0-100 numbers based on responses],
            "analysis": {{
                "Anxiety": {{"score": number, "rationale": "text"}},
                "Depression": {{"score": number, "rationale": "text"}},
                "Self-esteem": {{"score": number, "rationale": "text"}},
                "Stress": {{"score": number, "rationale": "text"}},
                "Sleep Quality": {{"score": number, "rationale": "text"}}
            }},
            "suggestions": {{
                "Immediate Actions": ["list", "of", "steps"],
                "Long-term Strategies": ["list", "of", "steps"]
            }},
            "well_being_tips": ["list", "of", "5+ tips"]
        }}
        
        - The `scores` should be calculated based on input data, **not predefined**.
        - The `analysis` should contain meaningful insights based on user responses.
        - **Do NOT include any extra text, explanations, or Markdown formatting.**
        - Having no Sleep issues indicate higher score.    
        - Steps to Overcome Issues Tell at least 4 of each.
        """

        # Generate response with timeout
        response = model.generate_content(prompt, request_options={"timeout": 15})
        
        if not response.text:
            print("⚠️ Empty Gemini response")
            return DEFAULT_ANALYSIS

        # Clean and parse response
        raw_json = response.text.strip().replace("```json", "").replace("```", "")
        try:
            analysis_data = json.loads(raw_json)
        except json.JSONDecodeError:
            print("❌ Invalid JSON response")
            return DEFAULT_ANALYSIS

        # Validate response structure
        required_keys = {
            "parameters", "scores", "analysis",
            "suggestions", "well_being_tips"
        }
        if not all(key in analysis_data for key in required_keys):
            print("⚠️ Invalid response structure")
            return DEFAULT_ANALYSIS

        # Validate array lengths
        if len(analysis_data["parameters"]) != 5 or len(analysis_data["scores"]) != 5:
            print("⚠️ Invalid parameter/scores length")
            return DEFAULT_ANALYSIS

        return analysis_data

    except Exception as e:
        print(f"❌ Analysis Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return DEFAULT_ANALYSIS

@app.route('/mental_health_report') 
def mental_health_report():
    # Check authentication
    email = session.get("email")
    if not email:
        flash("Please log in to view your report.", "danger")
        return redirect(url_for('login'))

    # Get user-specific file paths
    user_folder = get_user_folder(email)
    close_file = os.path.join(user_folder, "close_end_questions_responses.csv")
    open_file = os.path.join(user_folder, "open_end_questions_responses.csv")

    # Validate response files
    if not all(os.path.exists(f) for f in [close_file, open_file]):
        flash("Complete both assessments to generate report.", "warning")
        return redirect(url_for('session_page'))

    # Generate analysis with user-specific data
    try:
        analysis = send_to_gemini(close_file, open_file)
        analysis = analysis if isinstance(analysis, dict) else DEFAULT_ANALYSIS
        session["analysis"] = analysis
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        analysis = DEFAULT_ANALYSIS

    # Get user details
    user = User.query.filter_by(email=email).first()
    user_name = user.name if user else "User"
    user_email = user.email if user else "N/A"

    # Prepare Q&A data
    qa_data = []
    try:
        qa_data.extend([{"question": row[0], "answer": row[1]} 
                      for row in pd.read_csv(close_file).itertuples(index=False)])
        qa_data.extend([{"question": row[0], "answer": row[1]} 
                      for row in pd.read_csv(open_file).itertuples(index=False)])
    except Exception as e:
        print(f"Error loading Q&A data: {str(e)}")

    # Email sending logic
    if user and user.email != "N/A":
        try:
            # Generate PDF report
            pdf_path = generate_pdf(user_name, user_email, analysis)
            
            # Send email with PDF attachment
            success, message = send_email_with_pdf(user_email, pdf_path)
            
            if success:
                flash("Report has been sent to your email!", "success")
            else:
                flash(f"Email could not be sent: {message}", "warning")
                
        except Exception as e:
            print(f"Email sending error: {str(e)}")
            flash("Error sending email report", "danger")

    return render_template(
        "mental_health_report.html", 
        analysis=analysis,
        qa_data=qa_data,
        user_name=user_name,
        user_email=user_email,
        DEFAULT_ANALYSIS=DEFAULT_ANALYSIS
    )

def send_analysis_email(user_email, user_name, analysis_data):
    """Sends the mental health analysis email"""
    try:
        if not user_email:
            print("⚠️ No email address provided")
            return False

        msg = Message(
            "Your WeCare Mental Health Analysis",
            recipients=[user_email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )

        # Create text version
        msg.body = f"""
        Hi {user_name},

        Your Mental Health Analysis Report is ready.
        Key Insights:
        - Overall Score: {analysis_data.get('overall_score', 'N/A')}
        - Main Strengths: {', '.join(analysis_data.get('strengths', []))}
        - Recommendations: {', '.join(analysis_data.get('recommendations', []))}

        Download full report: [URL_TO_REPORT]
        """

        # Create HTML version
        msg.html = render_template(
            'analysis_email.html',
            user_name=user_name,
            analysis=analysis_data
        )

        # Add debugging
        print(f"📤 Attempting to send email to {user_email}")
        mail.send(msg)
        print("✅ Email sent successfully")
        return True

    except Exception as e:
        print(f"❌ Email send failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def format_analysis_text(analysis_dict):
    """Formats analysis dictionary into readable text."""
    if not analysis_dict:
        return "No analysis available."
    
    formatted = []
    for key, value in analysis_dict.items():
        if isinstance(value, dict):
            # For nested analysis (like {'Anxiety': {'score': 50, 'rationale': '...'}})
            rationale = value.get('rationale', 'No details available.')
            formatted.append(f"• {key}: {rationale}")
        else:
            formatted.append(f"• {key}: {value}")
    
    return "\n".join(formatted)

def format_wellbeing_tips(tips_list):
    """Formats wellbeing tips list into readable text."""
    if not tips_list:
        return "No tips available."
    return "\n".join(f"• {tip}" for tip in tips_list)

def send_email_with_pdf(recipient_email, pdf_path):
    """Sends email with PDF attachment"""
    try:
        msg = Message(
            "Your Mental Health Analysis Report",
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=[recipient_email]
        )
        msg.body = f"Please find your mental health analysis report attached."
        
        with app.open_resource(pdf_path) as fp:
            msg.attach(
                "mental_health_report.pdf",
                "application/pdf",
                fp.read()
            )
        
        mail.send(msg)
        return True, "Email sent successfully"
        
    except Exception as e:
        return False, str(e)

def generate_pdf(user_name, user_email, analysis):
    """Generates the PDF report server-side with radar chart."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # --------------------------
    # 1. Header Section
    # --------------------------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Mental Health Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # User details
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"User: {user_name}", ln=True, align="L")
    pdf.cell(200, 10, f"Email: {user_email}", ln=True, align="L")
    pdf.ln(15)

    # --------------------------
    # 2. Generate Radar Chart
    # --------------------------
    if analysis and "parameters" in analysis and "scores" in analysis:
        # Create radar chart image
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot data
        params = analysis["parameters"]
        scores = analysis["scores"]
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        
        ax.plot(angles, scores, color='#4a90e2', linewidth=2)
        ax.fill(angles, scores, color='#4a90e2', alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
        plt.ylim(0, 100)
        plt.xticks(angles, params, fontsize=12)
        
        # Save chart to temp file
        chart_path = "static/temp_radar.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        # Add chart to PDF
        pdf.image(chart_path, x=45, w=120)
        pdf.ln(10)

    # --------------------------
    # 3. Detailed Analysis
    # --------------------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Detailed Analysis", ln=True)
    pdf.ln(5)

    if analysis.get("analysis"):
        pdf.set_font("Arial", size=12)
        for param, data in analysis["analysis"].items():
            pdf.cell(200, 10, f"{param}: {data.get('score', 'N/A')}", ln=True)
            pdf.multi_cell(190, 6, data.get("rationale", "No rationale provided."))
            pdf.ln(2)

    # --------------------------
    # 4. Well-Being Tips
    # --------------------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Well-Being Tips", ln=True)
    pdf.ln(5)

    if analysis.get("well_being_tips"):
        pdf.set_font("Arial", size=12)
        for tip in analysis["well_being_tips"]:
            pdf.multi_cell(190, 6, f"- {tip}")
            pdf.ln(2)

    # --------------------------
    # 5. Save PDF
    # --------------------------
    pdf_path = "static/reports/Mental_Health_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    # Expect the PDF file (blob) in the form field "pdf"
    pdf_file = request.files.get('pdf')
    recipient_email = request.form.get('email')

    if not pdf_file:
        return jsonify({"success": False, "message": "No PDF file received"}), 400

    pdf_path = "static/reports/Mental_Health_Report.pdf"
    # Ensure the directory exists; create if it does not.
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    # Save the PDF (overwrite if it already exists)
    pdf_file.save(pdf_path)

    # If an email address is provided, send the saved PDF via email.
    if recipient_email:
        email_status = send_email_with_pdf(recipient_email, pdf_path)
    else:
        email_status = "No email provided"

    return jsonify({"success": True, "message": "PDF saved", "status": email_status})

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    try:
        # Get user details from session
        user_name = session.get("user_name", "User")
        user_email = session.get("email", "No Email")
        analysis = session.get("analysis", {})
        
        # Generate PDF
        pdf_path = generate_pdf(user_name, user_email, analysis)
        
        # Check if email is requested
        recipient_email = request.args.get("email")
        if recipient_email:
            # Validate email format
            if "@" not in recipient_email or "." not in recipient_email:
                return jsonify({"success": False, "message": "Invalid email address"})
            
            # Send email
            success, message = send_email_with_pdf(recipient_email, pdf_path)
            if not success:
                return jsonify({"success": False, "message": message})
        
        # Send PDF as download
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="Mental_Health_Report.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/send_pdf_email", methods=["POST"])
def send_pdf_email():
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({"success": False, "message": "Missing JSON in request"}), 400
            
        data = request.get_json()
        if "email" not in data:
            return jsonify({"success": False, "message": "Missing email address"}), 400
            
        # Get user details
        email = session.get("email")
        if not email:
            return jsonify({"success": False, "message": "User not authenticated"}), 401
            
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
            
        # Generate PDF
        pdf_path = generate_pdf(user.name, user.email, session.get("analysis", {}))
        
        # Send email
        success, message = send_email_with_pdf(data["email"], pdf_path)
        if not success:
            return jsonify({"success": False, "message": message}), 500
            
        return jsonify({"success": True, "message": "Email sent successfully"})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    

# Forgot Password Route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate a random 4-digit OTP
            otp = "{:04d}".format(random.randint(0, 9999))
            # OTP is valid for 5 minutes (300 seconds)
            expiration = time.time() + 300  
            # Store OTP details in session
            session['reset_otp'] = otp
            session['otp_expiration'] = expiration
            session['reset_email'] = email

            # Send OTP email
            msg = Message("Your OTP for Password Reset", recipients=[email])
            msg.body = f"Your OTP for resetting your password is {otp}. It is valid for 5 minutes."
            mail.send(msg)
            flash("An OTP has been sent to your email address.", "success")
            return redirect(url_for('reset_password'))
        else:
            flash("Email address not found. Please check and try again.", "danger")
    return render_template('forgot_password.html')

# Reset Password Route
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        otp_entered = request.form['otp']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return redirect(url_for('reset_password'))

        stored_otp = session.get('reset_otp')
        expiration = session.get('otp_expiration')
        reset_email = session.get('reset_email')

        # Check if OTP session data exists
        if not stored_otp or not expiration or not reset_email:
            flash("Session expired. Please try again.", "danger")
            return redirect(url_for('forgot_password'))

        # Check if the OTP has expired
        if time.time() > expiration:
            flash("OTP has expired. Please request a new one.", "danger")
            session.pop('reset_otp', None)
            session.pop('otp_expiration', None)
            session.pop('reset_email', None)
            return redirect(url_for('forgot_password'))

        # Validate the entered OTP
        if otp_entered != stored_otp:
            flash("Invalid OTP. Please try again.", "danger")
            return redirect(url_for('reset_password'))

        # OTP is valid; update user's password
        user = User.query.filter_by(email=reset_email).first()
        if user:
            user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            db.session.commit()
            flash("Password reset successfully. Please log in with your new password.", "success")
            # Clear OTP session data
            session.pop('reset_otp', None)
            session.pop('otp_expiration', None)
            session.pop('reset_email', None)
            return redirect(url_for('login'))
        else:
            flash("User not found.", "danger")
            return redirect(url_for('forgot_password'))
    return render_template('reset_password.html')

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    otp_entered = request.form['otp']
    stored_otp = session.get('reset_otp')
    
    if not stored_otp:
        return jsonify({'valid': False, 'message': 'OTP session expired'})
    
    if time.time() > session.get('otp_expiration', 0):
        return jsonify({'valid': False, 'message': 'OTP has expired'})
    
    if otp_entered != stored_otp:
        return jsonify({'valid': False, 'message': 'Invalid OTP'})
    
    return jsonify({'valid': True})

@app.route('/resend_otp')
def resend_otp():
    email = session.get('reset_email')
    if not email:
        return jsonify({'success': False})
    
    # Generate new OTP
    otp = "{:04d}".format(random.randint(0, 9999))
    expiration = time.time() + 300
    
    # Update session
    session['reset_otp'] = otp
    session['otp_expiration'] = expiration
    
    # Send email
    msg = Message("New OTP for Password Reset", recipients=[email])
    msg.body = f"Your new OTP is {otp}. Valid for 5 minutes."
    mail.send(msg)
    
    return jsonify({'success': True})


def get_user_folder(email):
    user_folder = os.path.join("responses", email)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder


@app.route('/current_mood')
def get_current_mood():
    return jsonify({
        'mood': session.get('last_mood', 'neutral')
    })
@app.route('/generate_encouragement', methods=['POST'])
def generate_encouragement():
    global dominant_emotion  # Access the emotion detection from your camera system
    
    prompt = f"""**User Emotion**: {dominant_emotion.lower()}

Generate a conversational engagement message following these rules:
1. **Content Requirements**:
   - Length: 12-15 words exactly
   - Tone: Mirror {dominant_emotion.lower()} emotion subtly ({"soothing" if dominant_emotion.lower() in ['sad', 'angry'] else "enthusiastic"})
   - Structure: Open-ended question or gentle invitation to share
   - Never use phrases like "love to hear" or "I notice"

2. **Emotion Handling**:
   {f"Avoid direct mood references" if dominant_emotion.lower() == "neutral" else f"Acknowledge but don't name the emotion"}
   - Neutral emotion example: "Beautiful day to share thoughts! What's on your mind? 🌸"
   - Non-neutral example: "Would a listening ear help right now? I'm here 💬"

3. **Stylistic Guidelines**:
   - Use casual, conversational language
   - Include exactly 1 relevant emoji at end
   - Avoid exclamation marks for negative emotions
   - Focus on creating safe space for sharing

Final output must be only the message text with emoji, no prefixes or explanations."""

    try:
        # Configure model with emotion-aware settings
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "max_output_tokens": 60,
                "temperature": 0.8 if dominant_emotion != "Neutral" else 0.5
            }
        )
        
        # Generate response with emotional context
        response = model.generate_content(prompt, request_options={"timeout": 7})
        
        return jsonify({'message': response.text.strip()})
    
    except Exception as e:
        print(f"Encouragement generation error: {str(e)}")
        # Fallback messages based on detected emotion
        return jsonify({'message': 
            f"I sense you might be feeling {dominant_emotion.lower()}... I'm here to listen 🌟" 
            if dominant_emotion != "Neutral" else 
            "Let me know if you'd like to talk about anything 💭"
        })

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    email = session['email']
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    if not user.check_password(current_password):
        return jsonify({'success': False, 'error': 'current_password', 'message': 'Incorrect current password'}), 401
    
    try:
        # Update password
        user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.session.commit()
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    email = session['email']
    password = request.form.get('password')
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    if not user.check_password(password):
        return jsonify({'success': False, 'message': 'Incorrect password'}), 401
    
    try:
        # Delete user responses folder if exists
        user_folder = get_user_folder(email)
        if os.path.exists(user_folder):
            import shutil
            shutil.rmtree(user_folder)
        
        # Delete user from database
        db.session.delete(user)
        db.session.commit()
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Account deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
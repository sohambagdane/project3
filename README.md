<<<<<<< HEAD
# WeCare - Mental Health Support Platform

A Flask-based web application that provides mental health support through various features including emotion detection, chat support, and mental health analysis.

## Features

- User Authentication (Login/Register)
- Emotion Detection using OpenCV
- Chat Support with Gemini AI
- Mental Health Analysis
- PDF Report Generation
- Email Notifications
- Open and Close-ended Questionnaires

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hxrshexe/WeCare.git
cd WeCare
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your environment variables:
```
SECRET_KEY=your-secret-key
API_KEY=your-gemini-api-key
MAIL_PASSWORD=your-mail-password
```

5. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Running the Application

1. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
WeCare/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── static/               # Static files (CSS, JS, images)
│   ├── uploads/          # User uploads
│   └── reports/          # Generated reports
├── templates/            # HTML templates
├── dataset/             # Training data
└── responses/           # User responses
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask Framework
- OpenCV for face detection
- Google Gemini AI for chat support
- FPDF for PDF generation 
=======
# project
>>>>>>> 068220559657ac5777f213f0470920ca3f60c5ed

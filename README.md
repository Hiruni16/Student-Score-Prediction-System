# Student Score Predictor 🎓

An AI-powered web application that predicts student exam scores using a deep learning model. This project was developed as part of the **Computational Intelligence (CIS6005)** module at Cardiff Metropolitan University.

## 📌 Project Overview

The Student Score Predictor uses an Artificial Neural Network (ANN) to estimate students' exam scores based on academic, demographic, and behavioral factors.

The application allows users to create accounts, log in, enter student information, and receive instant exam score predictions through a user-friendly web interface.

## ✨ Features

- **User Registration & Login** – Secure user authentication and session management.
- **Exam Score Prediction** – Predicts exam scores using a trained ANN model.
- **Student Data Input** – Accepts academic and behavioral information.
- **Real-Time Results** – Displays predicted scores immediately after submission.
- **Input Validation** – Validates user inputs and displays appropriate error messages.
- **Responsive Interface** – User-friendly web interface for interacting with the system.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Frontend | HTML5, CSS3, Jinja2 |
| Backend | Python, FastAPI |
| Machine Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Database | MySQL |
| Development Environment | Google Colab, VS Code |
| Dataset | Kaggle – Predicting Student Test Scores |

## 🤖 Machine Learning Model

The project uses a feed-forward Artificial Neural Network for regression.

The model:
- Learns complex relationships between student attributes and exam scores.
- Uses numerical feature scaling and categorical one-hot encoding.
- Predicts exam scores on a scale of 0–100.
- Was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and prediction trend validation.

## 📊 Input Features

The prediction form includes:

- Age
- Gender
- Course
- Daily study hours
- Class attendance percentage
- Sleep duration
- Sleep quality
- Study method
- Internet access
- Facility rating
- Exam difficulty level

## 🏗️ System Architecture

The application follows a three-tier architecture:

1. **Presentation Layer:** HTML, CSS, and Jinja2 templates for user interaction.
2. **Application Layer:** FastAPI handles authentication, validation, and prediction requests.
3. **Data & Machine Learning Layer:** MySQL stores user information, while the trained ANN generates predictions.

### Prediction Workflow

1. User registers or logs in.
2. User enters academic and behavioral information.
3. The backend validates and preprocesses the input.
4. The trained ANN predicts the exam score.
5. The prediction result is displayed to the user.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- MySQL Server
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd <YOUR_PROJECT_FOLDER>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```powershell
   venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the MySQL database and update the database connection settings in `db.py`.

5. Ensure the trained ANN model and preprocessing files are available at the paths expected by the application.

6. Start the FastAPI application using the project's entry point. For example, if the entry point is `app.py` and the FastAPI instance is named `app`:

   ```bash
   uvicorn app:app --reload
   ```

7. Open the local URL displayed in the terminal to access the application.


## 📁 Project Structure

```text
Student-Score-Predictor/
├── app.py                 # FastAPI application
├── db.py                  # MySQL database connection
├── templates/
│   ├── home.html          # Home page
│   ├── index.html         # Prediction interface
│   ├── register.html      # Registration page
│   └── login.html         # Login page
├── static/                 # CSS and static assets
├── model/                  # Trained model and preprocessing files
├── requirements.txt        # Python dependencies
└── README.md
```

## 🎯 Project Objectives

- Apply deep learning techniques to educational data.
- Predict student academic performance using an ANN.
- Develop a practical, web-based AI application.
- Demonstrate the integration of machine learning, backend development, and database technologies.

## 👩‍💻 Author

**D.M. Chathurya Hiruni Gunathilaka**


## 📚 Dataset

[Kaggle – Predicting Student Test Scores](https://www.kaggle.com/competitions/playground-series-s6e1)


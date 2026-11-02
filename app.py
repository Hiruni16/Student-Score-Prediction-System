import joblib
import pandas as pd
import numpy as np
import traceback
import re
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from starlette.middleware.sessions import SessionMiddleware
from db import get_db
import warnings
warnings.filterwarnings('ignore')


# ================ MODEL TRAINING FUNCTION ================
def train_grade_prediction_model():
    """Train a model with realistic diminishing returns on study hours"""
    print("\n" + "="*70)
    print("TRAINING GRADE PREDICTION MODEL")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 2000
    
    print(f"\nGenerating {n_samples} training samples...")
    
    age = np.random.randint(18, 26, n_samples)
    gender = np.random.choice(['male', 'female'], n_samples)
    course = np.random.choice(['Computer Science', 'Engineering', 'Commerce', 'Arts', 'Medicine'], n_samples)
    
    study_hours = np.random.uniform(1, 16, n_samples)
    class_attendance = np.random.uniform(30, 100, n_samples)
    
    internet_access = np.random.choice(['yes', 'no'], n_samples)
    sleep_hours = np.random.uniform(4, 10, n_samples)
    sleep_quality = np.random.choice(['poor', 'average', 'good'], n_samples)
    study_method = np.random.choice(['self-study', 'online videos', 'group study', 'coaching'], n_samples)
    facility_rating = np.random.choice(['low', 'medium', 'high'], n_samples)
    exam_difficulty = np.random.choice(['easy', 'moderate', 'hard'], n_samples)
    
    print("Calculating grades with realistic diminishing returns...")
    
    base_grade = 20
    
    # PRIMARY FACTOR 1: Study Hours with diminishing and eventual decline
    # 1h ≈ 8, 8h ≈ 64, 12h ≈ 80, 16h ≈ 68
    study_impact = np.zeros(n_samples)
    for i, sh in enumerate(study_hours):
        if sh <= 8:
            # Efficient gains up to 8 hours
            study_impact[i] = sh * 8
        elif sh <= 12:
            # Slower gains from 8 to 12 hours
            study_impact[i] = 64 + (sh - 8) * 4
        else:
            # Fatigue/inefficiency: decline beyond 12 hours
            study_impact[i] = 80 - (sh - 12) * 3
    
    # PRIMARY FACTOR 2: Attendance (linear, 0-35 points)
    attendance_impact = (class_attendance - 30) / 70 * 35
    
    # SECONDARY FACTORS - HIGH VARIABILITY
    sleep_quality_impact = np.where(sleep_quality == 'good', 15,
                                   np.where(sleep_quality == 'average', 8, 0))
    internet_impact = np.where(internet_access == 'yes', 5, 0)
    sleep_impact = np.where((sleep_hours >= 6) & (sleep_hours <= 8), 6,
                           np.where((sleep_hours >= 5) & (sleep_hours <= 9), 3, 0))
    facility_impact = np.where(facility_rating == 'high', 8,
                              np.where(facility_rating == 'medium', 4, 0))
    study_method_impact = np.where(study_method == 'coaching', 15,
                                  np.where(study_method == 'group study', 8,
                                          np.where(study_method == 'online videos', 4, 0)))
    difficulty_impact = np.where(exam_difficulty == 'easy', 18,
                                np.where(exam_difficulty == 'moderate', 0, -18))
    
    # COMBINE - with high noise for unpredictability
    final_grade = (base_grade + 
                   study_impact + 
                   attendance_impact + 
                   sleep_quality_impact + 
                   internet_impact + 
                   sleep_impact + 
                   facility_impact + 
                   study_method_impact + 
                   difficulty_impact +
                   np.random.normal(0, 4, n_samples))
    
    # Soft cap near 95 to avoid saturation at a single value
    final_grade = np.maximum(final_grade, 0)
    final_grade = np.where(final_grade <= 95, final_grade, 95 - (final_grade - 95) * 0.05)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'course': course,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'internet_access': internet_access,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating,
        'exam_difficulty': exam_difficulty,
        'score': final_grade
    })
    
    print("\n" + "="*70)
    print("DATA ANALYSIS")
    print("="*70)
    
    study_corr = data['study_hours'].corr(data['score'])
    attendance_corr = data['class_attendance'].corr(data['score'])
    
    print(f"Study Hours ↔ Grade:    {study_corr:.4f} {'✓ STRONG' if study_corr > 0.7 else ''}")
    print(f"Attendance ↔ Grade:     {attendance_corr:.4f} {'✓ STRONG' if attendance_corr > 0.7 else ''}")
    print(f"Grade range: {data['score'].min():.1f} - {data['score'].max():.1f}")
    
    X = data.drop('score', axis=1)
    y = data['score']
    
    X = pd.get_dummies(X, columns=['gender', 'course', 'internet_access', 'sleep_quality',
                                     'study_method', 'facility_rating', 'exam_difficulty'],
                        drop_first=False)
    
    columns = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    print("\n" + "="*70)
    print("BUILDING & TRAINING MODEL")
    print("="*70)
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.15,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f}")
    
    print("\n" + "="*70)
    print("REALISTIC PREDICTIONS (Diminishing Returns)")
    print("="*70)
    
    def test_pred(sh, ap, sq='good', sm='group study', ed='moderate'):
        test_df = pd.DataFrame({
            'age': [20],
            'study_hours': [sh],
            'class_attendance': [ap],
            'sleep_hours': [7],
            'gender': ['male'],
            'course': ['Computer Science'],
            'internet_access': ['yes'],
            'sleep_quality': [sq],
            'study_method': [sm],
            'facility_rating': ['medium'],
            'exam_difficulty': [ed]
        })
        test_df = pd.get_dummies(test_df)
        test_df = test_df.reindex(columns=columns, fill_value=0)
        pred = model.predict(scaler.transform(test_df), verbose=0)[0][0]
        # Soft cap near 95 to keep variation above 10h
        pred = max(pred, 0)
        if pred > 95:
            pred = 95 - (pred - 95) * 0.05
        return pred
    
    print("\nWith same attendance (80%) and good conditions:\n")
    print("Study Hours | Grade")
    print("-" * 25)
    for sh in [2, 4, 6, 8, 10, 12, 14, 16]:
        grade = test_pred(sh, 80)
        print(f"{sh:2}h        | {grade:5.1f}/100")
    
    print("\n" + "-"*70)
    print("✓ Notice: More hours help up to ~12h, then decline")
    print("✓ 14h study does NOT guarantee high score due to:")
    print("   - Exam difficulty variation (±18 points)")
    print("   - Sleep quality variation (±15 points)")
    print("   - Random noise (±4 points)")
    print("✓ Maximum possible: ~95/100 with realistic cap")
    print("-"*70)
    
    model.save("exam_score_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(columns, "columns.pkl")
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINED AND SAVED")
    print("="*70 + "\n")
    
    return model, scaler, columns


# ================ APP INITIALIZATION ================
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="secret-key"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    if os.path.exists("exam_score_model.h5") and os.path.exists("scaler.pkl") and os.path.exists("columns.pkl"):
        print("Loading existing model...")
        model = load_model("exam_score_model.h5", compile=False)
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        print("✓ Model loaded successfully\n")
    else:
        print("Model files not found. Training new model...")
        model, scaler, columns = train_grade_prediction_model()
except Exception as e:
    print(f"Error loading model: {e}. Training new model...")
    model, scaler, columns = train_grade_prediction_model()


# ================ ENDPOINTS ================

@app.get("/admin/retrain")
async def retrain():
    """Retrain the model with fresh data"""
    try:
        global model, scaler, columns
        model, scaler, columns = train_grade_prediction_model()
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    success = request.session.pop("success", None)
    error = request.session.pop("error", None)
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "success": success, "error": error}
    )


@app.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if password != confirm_password:
        request.session["error"] = "Passwords do not match"
        return RedirectResponse("/register", status_code=302)

    if len(password) < 6:
        request.session["error"] = "Password must be at least 6 characters"
        return RedirectResponse("/register", status_code=302)

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        request.session["error"] = "Invalid email format"
        return RedirectResponse("/register", status_code=302)

    if not name.strip():
        request.session["error"] = "Name cannot be empty"
        return RedirectResponse("/register", status_code=302)

    db = None
    cur = None

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute("SELECT id FROM users WHERE email=%s", (email.lower(),))
        if cur.fetchone():
            request.session["error"] = "Email already registered"
            return RedirectResponse("/register", status_code=302)

        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name.strip(), email.lower(), password)
        )
        db.commit()
        request.session["success"] = "Registration successful! You can now login."
        return RedirectResponse("/register", status_code=302)

    except Exception:
        return HTMLResponse(f"<pre>{traceback.format_exc()}</pre>")

    finally:
        if cur:
            cur.close()
        if db:
            db.close()


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    error = request.session.pop("error", None)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": error}
    )


@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    db = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT * FROM users WHERE email=%s", (email.lower(),))
    user = cur.fetchone()

    cur.close()
    db.close()

    if not user or password != user["password"]:
        request.session["error"] = "Invalid email or password"
        return RedirectResponse("/login", status_code=302)

    request.session["user_id"] = user["id"]
    return RedirectResponse("/predict", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


@app.get("/admin/users")
async def list_users():
    db = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT id, name, email FROM users")
    users = cur.fetchall()

    cur.close()
    db.close()
    return {"users": users}


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    if "user_id" not in request.session:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    if "user_id" not in request.session:
        return RedirectResponse("/login", status_code=302)

    db = get_db()
    cur = db.cursor(dictionary=True)

    try:
        cur.execute(
            "SELECT id, age, predicted_score FROM predictions WHERE user_id=%s ORDER BY id DESC",
            (request.session["user_id"],)
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        db.close()

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "predictions": rows}
    )


@app.post("/history/delete", response_class=HTMLResponse)
async def delete_prediction(request: Request, prediction_id: int = Form(...)):
    if "user_id" not in request.session:
        return RedirectResponse("/login", status_code=302)

    db = get_db()
    cur = db.cursor()

    try:
        # Only delete if the record belongs to the logged-in user
        cur.execute(
            "DELETE FROM predictions WHERE id=%s AND user_id=%s",
            (prediction_id, request.session["user_id"]) 
        )
        db.commit()
    finally:
        cur.close()
        db.close()

    return RedirectResponse("/history", status_code=302)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    if "user_id" not in request.session:
        return RedirectResponse("/login", status_code=302)

    try:
        form = await request.form()
        data = dict(form)

        age = int(data["age"])
        data["study_hours"] = float(data["study_hours"])
        data["class_attendance"] = float(data["class_attendance"])
        data["sleep_hours"] = float(data["sleep_hours"])

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        score = float(model.predict(scaler.transform(df))[0][0])
        
        # Soft cap near 95 to avoid saturation while preventing 100%
        score = max(score, 0)
        if score > 95:
            score = 95 - (score - 95) * 0.05
        score = round(score, 2)

        db = get_db()
        cur = db.cursor()

        cur.execute(
            "INSERT INTO predictions (user_id, age, predicted_score) VALUES (%s, %s, %s)",
            (request.session["user_id"], age, score)
        )

        db.commit()
        cur.close()
        db.close()

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": score}
        )

    except Exception:
        return HTMLResponse(f"<pre>{traceback.format_exc()}</pre>")
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from passlib.hash import pbkdf2_sha256
from datetime import timedelta
import pandas as pd
import numpy as np
import joblib



# =========================================================
# LOAD AI MODEL
# =========================================================

model = joblib.load("models/scarcity_model33.pkl")


budget_model = joblib.load("budget_model.pkl")
# =========================================================
# APP INITIALIZATION
# =========================================================

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///nationwise.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "nationwise-super-secret-key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=2)

db = SQLAlchemy(app)
jwt = JWTManager(app)


# JWT ERROR HANDLERS


@jwt.unauthorized_loader
def missing_token(error):
    return jsonify({"error": "Authorization token missing"}), 401


@jwt.invalid_token_loader
def invalid_token(error):
    return jsonify({"error": "Invalid token"}), 401


@jwt.expired_token_loader
def expired_token(jwt_header, jwt_payload):
    return jsonify({"error": "Token expired"}), 401



# DATABASE MODELS
#  authentication DATA TABLE

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(120), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    role = db.Column(db.String(20), nullable=False)


   
    def set_password(self, password):
        self.password = pbkdf2_sha256.hash(password)


    def check_password(self, password):
        return pbkdf2_sha256.verify(password, self.password)


# BUDGET REPORT TABLE


class BudgetReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(120))
    sector = db.Column(db.String(120))
    allocated_budget = db.Column(db.Float)
    allocation_percentage = db.Column(db.Float)



# SCARCITY REPORT TABLE


class ScarcityReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(120))
    state = db.Column(db.String(120))
    resource = db.Column(db.String(120))
    scarcity_score = db.Column(db.Float)
    risk_zone = db.Column(db.String(50))



with app.app_context():
    db.create_all()


# AUTH ROUTES


@app.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "Username and password required"}), 400

    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "User already exists"}), 400

    user = User(
        username=data["username"],
        role=data.get("role", "viewer")
    )

    user.set_password(data["password"])

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request"}), 400

    user = User.query.filter_by(username=data.get("username")).first()

    if not user or not user.check_password(data.get("password")):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_access_token(identity=user.username)

    return jsonify({
        "access_token": token,
        "role": user.role
    })



# SAFE JSON HELPER


def safe(obj):

    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [safe(i) for i in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    return obj


# =========================================================
# AI BUDGET ALLOCATION
# =========================================================
@app.route("/upload-budget-excel", methods=["POST"])
@jwt_required()
def upload_budget_excel():

    current_user = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "Excel file not received"}), 400

    file = request.files["file"]

    # ---------------- TOTAL BUDGET ----------------
    try:
        total_budget_crore = float(request.form.get("total_budget", 0))

        if total_budget_crore <= 0:
            return jsonify({"error": "Total budget must be greater than 0"}), 400

    except:
        return jsonify({"error": "Invalid total budget"}), 400

    # ---------------- READ EXCEL ----------------
    try:
        df = pd.read_excel(file)

    except Exception as e:
        return jsonify({"error": f"Excel read error: {str(e)}"}), 400

    # ---------------- REQUIRED COLUMNS ----------------
    required_columns = ["Sector", "GDP", "Population", "Demand"]

    for col in required_columns:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    # ---------------- GROUP BY SECTOR ----------------
    df = df.groupby("Sector", as_index=False).sum()

    # ---------------- ADD INFLATION FEATURE ----------------
    df["Inflation"] = 6

    # ---------------- AI MODEL PREDICTION ----------------
    features = df[["GDP", "Population", "Demand", "Inflation"]]

    predicted_budget = budget_model.predict(features)

    df["predicted_budget"] = predicted_budget

    # ---------------- NORMALIZE TO TOTAL BUDGET ----------------
    total_budget_rupees = total_budget_crore * 1e7

    total_predicted = df["predicted_budget"].sum()

    if total_predicted == 0:
        df["allocation_ratio"] = 0
    else:
        df["allocation_ratio"] = df["predicted_budget"] / total_predicted

    df["Allocated Budget"] = df["allocation_ratio"] * total_budget_rupees
    df["allocation_percentage"] = df["allocation_ratio"] * 100

    # ---------------- SAVE RESULTS TO DATABASE ----------------
    for _, row in df.iterrows():

        report = BudgetReport(
            user=current_user,
            sector=row["Sector"],
            allocated_budget=float(row["Allocated Budget"]),
            allocation_percentage=float(row["allocation_percentage"])
        )

        db.session.add(report)

    db.session.commit()

    # ---------------- FIND TOP SECTOR ----------------
    top_sector = df.loc[df["Allocated Budget"].idxmax()].to_dict()

    # ---------------- RESPONSE ----------------
    return jsonify({
        "user": current_user,
        "total_budget_rupees": total_budget_rupees,
        "predictions": safe(df.to_dict(orient="records")),
        "top_sector": safe(top_sector)
    })

@app.route("/budget-history")
@jwt_required()
def budget_history():

    user = get_jwt_identity()

    reports = BudgetReport.query.filter_by(user=user).all()

    data = []

    for r in reports:
        data.append({
            "sector": r.sector,
            "budget": r.allocated_budget,
            "percentage": r.allocation_percentage
        })

    return jsonify(data)



# =========================================================
# AI RESOURCE SCARCITY INTELLIGENCE  (KMEANS)
# =========================================================
@app.route("/resource-scarcity", methods=["POST"])
@jwt_required()
def resource_scarcity():

    data = request.get_json()

    current_user = get_jwt_identity()

    state = data.get("state")
    resource = data.get("resource")
    availability = float(data.get("availability", 0))
    demand = float(data.get("demand", 0))

    if not state or not resource:
        return jsonify({"error": "State and resource are required"}), 400

    if availability <= 0:
        return jsonify({"error": "Availability must be greater than zero"}), 400


    # ---------------- AI SCARCITY CALCULATION ----------------

    demand_pressure = demand / availability
    population_factor = np.random.uniform(0.5, 1.5)

    # Predict cluster using trained AI model
    cluster = model.predict([[demand_pressure, population_factor]])[0]


    # ---------------- CLUSTER → RISK MAPPING ----------------

    if cluster == 0:
        zone = "Normal"
        actions = [
            "Routine monitoring",
            "Promote conservation awareness"
        ]

    elif cluster == 1:
        zone = "Watchlist"
        actions = [
            "Improve efficiency",
            "Demand-side management"
        ]

    else:
        zone = "Emergency"
        actions = [
            "Declare emergency",
            "Central government intervention"
        ]


    # ---------------- DRIVERS ----------------

    drivers = []

    if demand_pressure > 1:
        drivers.append("Demand exceeds available supply")

    if population_factor > 1:
        drivers.append("High population pressure")

    if not drivers:
        drivers.append("Supply and demand currently balanced")


    # ---------------- SCARCITY SCORE ----------------

    scarcity_score = (demand_pressure + population_factor) * 50


    # ---------------- SAVE TO DATABASE ----------------

    report = ScarcityReport(
        user=current_user,
        state=state,
        resource=resource,
        scarcity_score=float(scarcity_score),
        risk_zone=zone
    )

    db.session.add(report)
    db.session.commit()


    # ---------------- RESPONSE ----------------

    return jsonify({
        "state": state,
        "resource": resource,
        "scarcity_score": round(scarcity_score, 2),
        "risk_zone": zone,
        "key_drivers": drivers,
        "recommended_government_actions": actions
    })

@app.route("/scarcity-history")
@jwt_required()
def scarcity_history():

    user = get_jwt_identity()

    reports = ScarcityReport.query.filter_by(user=user).all()

    data = []

    for r in reports:
        data.append({
            "state": r.state,
            "resource": r.resource,
            "score": r.scarcity_score,
            "risk": r.risk_zone
        })

    return jsonify(data)



# =========================================================
# GOVERNMENT SCHEME RECOMMENDATION
# =========================================================

SCHEMES = {
    "Water": {
        "High": ["Jal Jeevan Mission", "National Water Mission"],
        "Medium": ["Atal Bhujal Yojana", "PMKSY Water Conservation"],
        "Low": ["Local Water Conservation Programs"]
    },
    "Electricity": {
        "High": ["UDAY Scheme", "National Smart Grid Mission"],
        "Medium": ["Power Sector Reforms", "Solar Rooftop Subsidy"],
        "Low": ["Renewable Energy Incentives"]
    },
    "Health": {
        "High": ["Ayushman Bharat", "National Health Mission"],
        "Medium": ["Health Infrastructure Upgrade"],
        "Low": ["Preventive Healthcare Awareness"]
    }
}

@app.route("/scheme-recommendation", methods=["POST"])
@jwt_required()
def scheme_recommendation():

    data = request.get_json()

    state = data.get("state")
    resource = data.get("resource")
    risk_level = data.get("risk_level")

    if not resource or not risk_level:
        return jsonify({"error": "Resource and risk level required"}), 400

    schemes = SCHEMES.get(resource, {}).get(risk_level, [])

    return jsonify({
        "state": state,
        "resource": resource,
        "risk_level": risk_level,
        "recommended_schemes": schemes
    })


# =========================================================
# 🕵️ RECRUITMENT FRAUD DETECTION
# =========================================================

@app.route("/recruitment-fraud", methods=["POST"])
@jwt_required()
def recruitment_fraud():

    data = request.get_json()

    department = data.get("department")
    applicants = int(data.get("applicants", 0))
    vacancies = int(data.get("vacancies", 1))
    selected = int(data.get("selected", 0))
    avg_score = float(data.get("exam_score_avg", 0))

    fraud_score = 0

    # Rule 1: Too many applicants per vacancy
    if applicants / vacancies > 150:
        fraud_score += 40

    # Rule 2: Very low average score
    if avg_score < 40:
        fraud_score += 30

    # Rule 3: Too few candidates selected
    if selected < vacancies * 0.7:
        fraud_score += 30

    # Risk Classification
    if fraud_score < 40:
        risk = "Low Risk"
    elif fraud_score < 70:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return jsonify({
        "department": department,
        "fraud_score": fraud_score,
        "risk_level": risk
    })





# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/")
def home():
    return jsonify({"status": "NationWise Backend Running"})


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
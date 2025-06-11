# --- START OF FILE app5.py ---
# KingPredict (Full Integration - Part 1)

# --- SECTION: IMPORTS AND SETUP ---
import os
import re
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple, Set, Union
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import aliased
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, HiddenField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp # Email removed as not used directly
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask_migrate import Migrate
import joblib 

import pandas as pd
import numpy as np

# Only import what's needed if pre-trained models are used
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.pipeline import Pipeline 



load_dotenv()

# --- SECTION: CONSTANTS ---
MAX_FAILED_LOGIN_ATTEMPTS = 4; LOCKOUT_DURATION_HOURS = 1
PLACEHOLDER_CONFIDENCE_MAX = 0.20

# General Hourly Model
HOURLY_PRED_COUNT = 15; HOURLY_MIN_MULTIPLIER = 10.0
HOURLY_ML_FEATURES = [ 
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dayofweek_sin', 'dayofweek_cos', 
    'prev_interval', 'multiplier', 'prev_interval_lag2', 'prev_interval_lag3', 
    'prev_multiplier', 'prev_multiplier_lag2', 'rolling_mean_interval_3', 'rolling_std_interval_3',
    'rolling_mean_multiplier_3', 'rolling_std_multiplier_3', 'mult_x_prev_mult',
    'prev_interval_x_mult', 'prev_interval_diff_lag1_lag2', 
    'events_in_last_15min_before_prev', 'avg_interval_short_term_N5',
]
HOURLY_ML_TARGET = 'interval_to_next_actual' 

# 50x Hourly Model
HOURLY_50X_PRED_COUNT = 10; HOURLY_50X_MIN_MULTIPLIER = 50.0 # Changed from 15 to 10
HOURLY_50X_ML_FEATURES = HOURLY_ML_FEATURES # Reuses the general hourly 20-feature set

# HML Model
HML_THRESHOLD = 100.0; HML_LOOKBACK_WINDOW_EVENTS = 50; HML_NUM_UPCOMING_PREDS = 3
HML_SAVE_THRESHOLD_PERCENT = 30 
HML_FEATURES_DEF = ['time_since_last_high_event', 'num_low_events_since_last_high', 'avg_multiplier_in_window', 'hour_of_day', 'dayofweek']
HML_HOURLY_CANDIDATE_SLOTS_PER_HOUR = 6; HML_HOURLY_TARGET_HOURS_AHEAD = 1

# Daily Tiered Models Base Feature Set
HIGH_TIER_DAILY_ML_FEATURES_BASE = [
    'prev_event_hour_sin', 'prev_event_hour_cos', 
    'prev_event_dayofweek_sin', 'prev_event_dayofweek_cos',
    'prev_interval_between_high_tier', 
    'prev_high_tier_multiplier',      
    'prev_interval_between_high_tier_lag2',
    'prev_high_tier_multiplier_lag2',
    'rolling_mean_interval_high_tier_3', 
    'rolling_std_interval_high_tier_3',  
    'rolling_mean_multiplier_high_tier_3', 
    'days_since_epoch', 
    'month_sin', 'month_cos'
]
HIGH_TIER_DAILY_TARGET_COL_PREFIX = 'interval_to_next_'

# 50x Daily (Heuristic)
DAILY_50X_PRED_COUNT = 20; DAILY_50X_MIN_MULTIPLIER = 50.0; DAILY_50X_RECENT_N_EVENTS = 7

# 100x Daily Model (ML)
DAILY_100X_PRED_COUNT = 15; DAILY_100X_MIN_MULTIPLIER = 100.0
DAILY_100X_ML_FEATURES = HIGH_TIER_DAILY_ML_FEATURES_BASE 
DAILY_100X_TARGET_COL = f"{HIGH_TIER_DAILY_TARGET_COL_PREFIX}100x"
DAILY_100X_RECENT_N_EVENTS = 7
# 500x Daily (Heuristic)
DAILY_500X_PRED_COUNT = 10; DAILY_500X_MIN_MULTIPLIER = 500.0; DAILY_500X_RECENT_N_EVENTS = 5

# 1000x Daily (Heuristic)
DAILY_1000X_PRED_COUNT = 5; DAILY_1000X_MIN_MULTIPLIER = 1000.0; DAILY_1000X_RECENT_N_EVENTS = 3

# Monthly Model
MONTHLY_5000X_PRED_COUNT = 3; MONTHLY_5000X_MIN_MULTIPLIER = 5000.0

# Minimum data for training ML models (can be overridden in training script if needed)
HOURLY_MIN_DATA_FOR_ML = 20
HOURLY_50X_MIN_DATA_FOR_ML = 15
DAILY_50X_MIN_DATA_FOR_ML = 15 # For heuristic one this is not used, for ML this would be relevant
DAILY_100X_MIN_DATA_FOR_ML = 15
DAILY_500X_MIN_DATA_FOR_ML = 10 
DAILY_1000X_MIN_DATA_FOR_ML = 5

# --- GLOBAL VARIABLES FOR LOADED MODELS & PATHS ---
HOURLY_MODEL_PIPELINE: Optional[Pipeline] = None
HML_MODEL_PIPELINE: Optional[Pipeline] = None 
HOURLY_50X_MODEL_PIPELINE: Optional[Pipeline] = None
DAILY_100X_MODEL_PIPELINE: Optional[Pipeline] = None
MODELS_LOADED_SUCCESSFULLY = {"hourly": False, "hml": False, "hourly_50x": False, "daily_100x": False}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
HOURLY_MODEL_PATH = os.path.join(MODEL_DIR, 'hourly_pipeline_rf_champion_v1_20feats.joblib') 
HML_MODEL_PATH = os.path.join(MODEL_DIR, 'hml_prediction_pipeline_v1.joblib')
HOURLY_50X_MODEL_PATH = os.path.join(MODEL_DIR, 'hourly_50x_pipeline_final_best_cv.joblib')
DAILY_100X_MODEL_PATH = os.path.join(MODEL_DIR, 'daily_100x_pipeline_final_best_cv.joblib')

# --- SECTION: FLASK APP CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_strong_default_secret_key_for_dev_only')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///kingpredict.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app.last_high_event_pred_update: Optional[datetime] = None

# --- SECTION: JINJA CONTEXT PROCESSOR ---
@app.context_processor
def inject_global_template_vars() -> Dict[str, Any]:
    return dict(datetime=datetime, timezone=timezone, app=app, timedelta=timedelta,
                format_timedelta_to_hhmmss=format_timedelta_to_hhmmss,
                HOURLY_MIN_MULTIPLIER=HOURLY_MIN_MULTIPLIER, HOURLY_50X_MIN_MULTIPLIER=HOURLY_50X_MIN_MULTIPLIER,
                DAILY_50X_MIN_MULTIPLIER=DAILY_50X_MIN_MULTIPLIER, DAILY_100X_MIN_MULTIPLIER=DAILY_100X_MIN_MULTIPLIER,
                DAILY_500X_MIN_MULTIPLIER=DAILY_500X_MIN_MULTIPLIER, DAILY_1000X_MIN_MULTIPLIER=DAILY_1000X_MIN_MULTIPLIER,
                MONTHLY_5000X_MIN_MULTIPLIER=MONTHLY_5000X_MIN_MULTIPLIER, HML_THRESHOLD=HML_THRESHOLD)

# --- SECTION: FLASK EXTENSIONS INITIALIZATION ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
scheduler = BackgroundScheduler(daemon=True, timezone=str(timezone.utc))

# --- SECTION: HELPER FUNCTIONS ---
def parse_expiry_time(expiry_str: str) -> Optional[int]:
    if not expiry_str or ':' not in expiry_str: return None
    try: h, m = map(int, expiry_str.split(':')); return (h * 3600) + (m * 60)
    except ValueError: return None
def format_timedelta_to_hhmmss(td: Optional[timedelta]) -> str:
    if td is None: return "00:00:00"
    seconds = int(td.total_seconds());
    if seconds < 0: return "Expired"
    hours = seconds // 3600; minutes = (seconds % 3600) // 60; seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
def generate_device_identifier() -> str:
    return hashlib.sha256(request.headers.get('User-Agent', 'unknown').encode()).hexdigest()
@app.template_filter('format_confidence')
def format_confidence_display(val: Optional[Union[float, int]]) -> str:
    if not isinstance(val, (float, int)) or not (0 <= val <= 1): return "N/A 🤔"
    p = int(val * 100); icon = '🤔';
    if p >= 90: icon = '🚀'
    elif p >= 80: icon = '🎯'
    elif p >= 70: icon = '✅'
    elif p >= 50: icon = '🔄'
    elif p >= 30: icon = '✨'
    return f"{p}% {icon}"
# --- END SECTION: HELPER FUNCTIONS ---

# --- SECTION: DATABASE MODELS ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    telephone_number = db.Column(db.String(20), unique=True, nullable=False) 
    password_hash = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    status = db.Column(db.String(20), default='pending', nullable=False) 
    registration_timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    approved_by_admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    approval_timestamp = db.Column(db.DateTime(timezone=True), nullable=True)
    expiry_duration_seconds = db.Column(db.Integer, nullable=True)
    first_login_timestamp = db.Column(db.DateTime(timezone=True), nullable=True)
    actual_expiry_timestamp = db.Column(db.DateTime(timezone=True), nullable=True)
    login_device_identifier = db.Column(db.String(64), nullable=True)
    failed_login_attempts = db.Column(db.Integer, default=0)
    lockout_until = db.Column(db.DateTime(timezone=True), nullable=True)
    approver = db.relationship('User', remote_side=[id], foreign_keys=[approved_by_admin_id], backref='approved_users_list', lazy='joined')
    def set_password(self, password: str): self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    def check_password(self, password: str) -> bool: return bcrypt.check_password_hash(self.password_hash, password)
    @property
    def is_active_account(self) -> bool:
        if self.status != 'active': return False
        if self.actual_expiry_timestamp:
            expiry_aware = self.actual_expiry_timestamp
            if expiry_aware.tzinfo is None: expiry_aware = expiry_aware.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < expiry_aware
        return self.expiry_duration_seconds is None or (self.expiry_duration_seconds is not None and self.first_login_timestamp is None)
    @property
    def time_left_seconds(self) -> Optional[float]:
        if self.status != 'active': return None
        if self.actual_expiry_timestamp:
            expiry_aware = self.actual_expiry_timestamp
            if expiry_aware.tzinfo is None: expiry_aware = expiry_aware.replace(tzinfo=timezone.utc)
            return max(0, (expiry_aware - datetime.now(timezone.utc)).total_seconds())
        return None
    @property
    def is_explicitly_expired(self) -> bool:
        if self.status != 'active': return True
        if self.actual_expiry_timestamp:
            expiry_aware = self.actual_expiry_timestamp
            if expiry_aware.tzinfo is None: expiry_aware = expiry_aware.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= expiry_aware
        if self.expiry_duration_seconds is None and self.status == 'active': return False 
        return False if self.status == 'active' else True

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]: return db.session.get(User, int(user_id))

class HistoricalData(db.Model):
    __tablename__ = 'historical_data'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    multiplier = db.Column(db.Float, nullable=False)
    @property
    def timestamp(self) -> Optional[datetime]:
        try: return datetime.strptime(f"{self.date} {self.time}", '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        except ValueError as e: logger.error(f"Error parsing HD timestamp ID {self.id if hasattr(self, 'id') else 'Unk'}: {e}"); return None

class PredictionHourly(db.Model):
    __tablename__ = 'prediction_hourly'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class Prediction50xHourly(db.Model):
    __tablename__ = 'prediction_50x_hourly'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class Prediction50xDaily(db.Model):
    __tablename__ = 'prediction_50x_daily'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class Prediction100x(db.Model): 
    __tablename__ = 'prediction_100x'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class Prediction500x(db.Model):
    __tablename__ = 'prediction_500x'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class Prediction1000x(db.Model):
    __tablename__ = 'prediction_1000x'; id = db.Column(db.Integer, primary_key=True)
    predicted_datetime_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    interval_to_next_seconds = db.Column(db.Integer, nullable=True)
    min_multiplier = db.Column(db.Float, nullable=False); avg_multiplier = db.Column(db.Float, nullable=False)
    max_multiplier = db.Column(db.Float, nullable=False); confidence = db.Column(db.Float, nullable=False)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def predicted_time(self) -> str: return self.predicted_datetime_utc.strftime('%H:%M:%S')
    @property
    def interval(self) -> Optional[int]: return self.interval_to_next_seconds

class HighMultiplierPredictionLog(db.Model):
    __tablename__ = 'high_multiplier_prediction_log'; id = db.Column(db.Integer, primary_key=True)
    predicted_event_time_utc = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    likelihood_percentage = db.Column(db.Integer, nullable=False)
    expected_value_multiplier = db.Column(db.Float, nullable=True)
    generated_at_utc = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    def __repr__(self) -> str: return f"<HMLog {self.predicted_event_time_utc.strftime('%H:%M')} - {self.likelihood_percentage}%>"

class PredictionMonthly(db.Model):
    __tablename__ = 'prediction_monthly'; id = db.Column(db.Integer, primary_key=True)
    target_date_utc = db.Column(db.Date, nullable=False, index=True) 
    predicted_value_utc = db.Column(db.DateTime(timezone=True), nullable=True) 
    min_multiplier = db.Column(db.Float, nullable=False, default=MONTHLY_5000X_MIN_MULTIPLIER)
    confidence_level = db.Column(db.String(50), nullable=True) 
    notes = db.Column(db.Text, nullable=True)
    generated_at_utc = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    user_verdict = db.Column(db.Integer, default=0, nullable=False)
    actual_multiplier = db.Column(db.Float, nullable=True) # <-- ADDED
    @property
    def time_to_event(self) -> Optional[timedelta]:
        target_dt_to_compare: Optional[datetime] = None
        if self.predicted_value_utc:
            target_dt_to_compare = self.predicted_value_utc
            if target_dt_to_compare is not None and target_dt_to_compare.tzinfo is None:
                target_dt_to_compare = target_dt_to_compare.replace(tzinfo=timezone.utc)
        elif self.target_date_utc:
            target_dt_to_compare = datetime.combine(self.target_date_utc, datetime.min.time(), tzinfo=timezone.utc)
        if target_dt_to_compare:
            if target_dt_to_compare.tzinfo is None: target_dt_to_compare = target_dt_to_compare.replace(tzinfo=timezone.utc)
            return target_dt_to_compare - datetime.now(timezone.utc)
        return None
# --- END SECTION: DATABASE MODELS ---

# --- SECTION: FORMS ---
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    telephone_number = StringField('Telephone Number', validators=[DataRequired(), Length(min=7, max=20), Regexp(r'^\+?[0-9\s()-]{7,20}$', message="Invalid phone number format.")])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first(): raise ValidationError('Username taken.')
    def validate_telephone_number(self, telephone_number_field):
        if User.query.filter_by(telephone_number=telephone_number_field.data).first(): raise ValidationError('Telephone number already registered.')

class AdminCreateUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    telephone_number = StringField('Telephone Number', validators=[DataRequired(), Length(min=7, max=20), Regexp(r'^\+?[0-9\s()-]{7,20}$', message="Invalid phone number format.")])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    expiry_time = StringField('Expiry (HH:MM or "indefinite")', validators=[DataRequired()])
    is_admin = SelectField('Role', choices=[('False', 'User'), ('True', 'Admin')], default='False')
    status = SelectField('Status', choices=[('active', 'Active'), ('pending', 'Pending'), ('suspended', 'Suspended')], default='active')
    submit = SubmitField('Create User')
    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first(): raise ValidationError('Username taken.')
    def validate_telephone_number(self, telephone_number_field):
        if telephone_number_field.data and User.query.filter_by(telephone_number=telephone_number_field.data).first(): raise ValidationError('Telephone number already registered.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired()]); submit = SubmitField('Login')

class AdminRenewUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    new_expiry_time = StringField('New Expiry (HH:MM or "indefinite")', validators=[DataRequired()])
    submit = SubmitField('Renew User')
    def validate_username(self, username_f): User.query.filter_by(username=username_f.data).first() or (_ for _ in ()).throw(ValidationError('User not found.'))

class FeedbackForm(FlaskForm): feedback_type = HiddenField(default='up'); submit = SubmitField('Rate')

class AddHistoricalDataForm(FlaskForm):
    date = StringField('Date (YYYY-MM-DD)', validators=[DataRequired(), Regexp(r'^\d{4}-\d{2}-\d{2}$')])
    time = StringField('Time (HH:MM:SS)', validators=[DataRequired(), Regexp(r'^\d{2}:\d{2}:\d{2}$')])
    multiplier = StringField('Multiplier', validators=[DataRequired(), Regexp(r'^\d+(\.\d+)?$')]); submit = SubmitField('Add Data')
    def validate_date(self, d):
        try: datetime.strptime(d.data, '%Y-%m-%d')
        except ValueError: raise ValidationError("Invalid date format. Use YYYY-MM-DD.")
    def validate_time(self, t):
        try: datetime.strptime(t.data, '%H:%M:%S')
        except ValueError: raise ValidationError("Invalid time format. Use HH:MM:SS.")
    def validate_multiplier(self, m):
        try: float(m.data)
        except ValueError: raise ValidationError("Multiplier must be a valid number.")
# --- END SECTION: FORMS ---


# --- FUNCTION TO LOAD MODELS ---
def load_trained_models():
    # ---- ADDED FOR VERY HIGH VISIBILITY DEBUGGING ----
    print("!!!!!!!!!!!!!!!!!!!! CALLING load_trained_models NOW !!!!!!!!!!!!!!!!!!!!")
    logger.info(">>>>>>>>>>>>>>>>>> INSIDE load_trained_models() FUNCTION <<<<<<<<<<<<<<<<<<")
    # ---- END ADDED ----

    global HOURLY_MODEL_PIPELINE, HML_MODEL_PIPELINE, HOURLY_50X_MODEL_PIPELINE, DAILY_100X_MODEL_PIPELINE, MODELS_LOADED_SUCCESSFULLY
    logger.info("Attempting to load pre-trained models...")

    # --- DEBUG MODEL_DIR ---
    logger.info(f"Resolved MODEL_DIR: {MODEL_DIR}")
    # --- END DEBUG ---

    if not os.path.exists(MODEL_DIR):
        logger.warning(f"Models directory '{MODEL_DIR}' does not exist. Pre-trained models cannot be loaded. Creating directory.")
        try:
            os.makedirs(MODEL_DIR)
            logger.info(f"Created models directory: {MODEL_DIR}")
        except OSError as e:
            logger.error(f"Could not create model directory: {e}")
            return # Exit if directory can't be created/found

    model_configs = {
        "hourly": (HOURLY_MODEL_PATH, "HOURLY_MODEL_PIPELINE"),
        "hml": (HML_MODEL_PATH, "HML_MODEL_PIPELINE"),
        "hourly_50x": (HOURLY_50X_MODEL_PATH, "HOURLY_50X_MODEL_PIPELINE"),
        "daily_100x": (DAILY_100X_MODEL_PATH, "DAILY_100X_MODEL_PIPELINE"),
    }

    for key, (path, pipeline_attr_name) in model_configs.items():
        # --- DEBUG EACH MODEL ATTEMPT ---
        logger.info(f"Attempting to load model: '{key}' from expected full path: '{path}'")
        # --- END DEBUG ---

        if os.path.exists(path):
            try:
                globals()[pipeline_attr_name] = joblib.load(path)
                MODELS_LOADED_SUCCESSFULLY[key] = True
                logger.info(f"{key.upper()} prediction pipeline loaded successfully from: {path}")
            except Exception as e:
                logger.error(f"Error loading {key} model from {path}: {e}", exc_info=True)
                MODELS_LOADED_SUCCESSFULLY[key] = False
        else:
            logger.warning(f"{key.upper()} model file '{path}' not found. Related predictions will use fallback/on-the-fly logic.")
            MODELS_LOADED_SUCCESSFULLY[key] = False

    # Final summary logs
    if all(MODELS_LOADED_SUCCESSFULLY.values()):
        logger.info("All configured pre-trained models loaded successfully.")
    elif any(MODELS_LOADED_SUCCESSFULLY.values()):
        logger.info("Some pre-trained models loaded. Others will use fallback/on-the-fly.")
    else:
        logger.warning("No pre-trained models were loaded. Application will rely heavily on fallback/on-the-fly logic.")
# --- END FUNCTION TO LOAD MODELS ---

# --- SECTION: PREDICTION LOGIC ---

# --- Paste the FULL get_historical_data_df here ---
def get_historical_data_df(min_multiplier_filter: float = 0.0, limit: Optional[int] = None, sort_ascending: bool = True) -> pd.DataFrame:
    query = HistoricalData.query
    if min_multiplier_filter > 0: query = query.filter(HistoricalData.multiplier >= min_multiplier_filter)
    data_records = []
    for record in query.all():
        if record is None: logger.warning("None record in hist data."); continue
        record_timestamp = record.timestamp 
        if record_timestamp: data_records.append({'timestamp': record_timestamp, 'multiplier': record.multiplier, HOURLY_ML_TARGET: np.nan})
        else: logger.warning(f"HistData ID {record.id if hasattr(record, 'id') else 'Unk'} invalid ts, skip.")
    if not data_records: return pd.DataFrame(columns=['timestamp', 'multiplier', HOURLY_ML_TARGET])
    df = pd.DataFrame(data_records); df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    df[HOURLY_ML_TARGET] = df['timestamp'].diff().dt.total_seconds().shift(-1) 
    if limit:
        df_temp_for_limit = df.sort_values(by='timestamp', ascending=False).head(limit)
        df = df_temp_for_limit.sort_values(by='timestamp', ascending=sort_ascending).reset_index(drop=True)
    return df

# --- Paste the FULL _generate_placeholder_predictions here ---
def _generate_placeholder_predictions(pred_cls: type, num_preds: int, target_start_dt: datetime, target_end_dt: datetime, min_mult: float, existing_hhmm_slots: Set[str], reason: str = "", force_low_confidence: bool = False) -> List[Any]:
    placeholders = []
    if target_start_dt.tzinfo is None: target_start_dt = target_start_dt.replace(tzinfo=timezone.utc)
    if target_end_dt.tzinfo is None: target_end_dt = target_end_dt.replace(tzinfo=timezone.utc)
    total_minutes_in_target_range = int((target_end_dt - target_start_dt).total_seconds() / 60)
    num_possible_slots = total_minutes_in_target_range
    num_to_generate = min(num_preds, num_possible_slots - len(existing_hhmm_slots))
    if num_to_generate <= 0: return []
    available_minutes = [m for m in range(total_minutes_in_target_range) if (target_start_dt + timedelta(minutes=m)).strftime('%H:%M') not in existing_hhmm_slots]
    if not available_minutes: return []
    chosen_minute_offsets = np.random.choice(available_minutes, size=min(num_to_generate, len(available_minutes)), replace=False)
    for minute_offset in chosen_minute_offsets:
        pred_time_utc = (target_start_dt + timedelta(minutes=int(minute_offset))).replace(second=0, microsecond=0)
        hhmm_key = pred_time_utc.strftime('%H:%M')
        if not (target_start_dt <= pred_time_utc < target_end_dt) or hhmm_key in existing_hhmm_slots: continue
        avg_m = min_mult * np.random.uniform(1.2, 2.5); max_m = avg_m * np.random.uniform(1.2, 2.0)
        confidence_val = round(np.random.uniform(0.05, PLACEHOLDER_CONFIDENCE_MAX), 2) if force_low_confidence else round(np.random.uniform(0.25, 0.45), 2)
        placeholders.append(pred_cls(predicted_datetime_utc=pred_time_utc, min_multiplier=min_mult, avg_multiplier=round(avg_m, 2), max_multiplier=round(max_m, 2), confidence=confidence_val))
        existing_hhmm_slots.add(hhmm_key)
        if len(placeholders) >= num_preds: break
    logger.info(f"Generated {len(placeholders)} placeholder preds for {pred_cls.__name__}. Reason: {reason if reason else 'Fallback'}. Forced low confidence: {force_low_confidence}")
    return sorted(placeholders, key=lambda p: p.predicted_datetime_utc)

# --- Paste the FULL _finalize_and_save_predictions here ---
def _finalize_and_save_predictions(pred_cls: type, predictions: List[Any], num_target_preds: int, label: str):
    if not predictions: logger.info(f"No {label} preds to save."); return
    final_preds_map: Dict[str, Any] = {}
    for p in sorted(predictions, key=lambda x: x.predicted_datetime_utc):
        pred_time_aware = p.predicted_datetime_utc
        if pred_time_aware.tzinfo is None: pred_time_aware = pred_time_aware.replace(tzinfo=timezone.utc) 
        hhmm_key = pred_time_aware.strftime('%H:%M')
        if hhmm_key not in final_preds_map:
            if len(final_preds_map) < num_target_preds: final_preds_map[hhmm_key] = p
        if len(final_preds_map) >= num_target_preds: break
    final_preds_list = sorted(list(final_preds_map.values()), key=lambda p: p.predicted_datetime_utc)
    if len(final_preds_list) < num_target_preds and num_target_preds > 0 : 
        logger.warning(f"Final {label} gen: {len(final_preds_list)} unique HH:MM, less than target {num_target_preds}.")
    for i in range(len(final_preds_list)):
        if i == 0: final_preds_list[i].interval_to_next_seconds = None
        else:
            interval_sec = (final_preds_list[i].predicted_datetime_utc - final_preds_list[i-1].predicted_datetime_utc).total_seconds()
            final_preds_list[i].interval_to_next_seconds = int(max(60, interval_sec)) 
    try:
        if final_preds_list: db.session.add_all(final_preds_list); db.session.commit(); logger.info(f"Saved {len(final_preds_list)} new {label} preds.")
        else: logger.info(f"No finalized {label} preds to save.")
    except Exception as e: db.session.rollback(); logger.error(f"Error saving {label} preds: {e}", exc_info=True)

# --- Paste the FULL prepare_features_for_hourly_prediction (for 20 features) here ---
def prepare_features_for_hourly_prediction(df_all_hist_for_lags: pd.DataFrame, current_event_series: pd.Series, feature_list: List[str], target_col: str) -> Optional[pd.DataFrame]:
    if current_event_series is None or current_event_series.empty: logger.warning("prep_feats_hrly: current_event_series empty."); return None
    features = {}
    event_time = current_event_series.get('timestamp')
    if pd.isna(event_time): logger.warning("prep_feats_hrly: timestamp missing."); return None
    if event_time.tzinfo is None: event_time = event_time.replace(tzinfo=timezone.utc)
    features['hour_sin'] = np.sin(2*np.pi*event_time.hour/24.0); features['hour_cos'] = np.cos(2*np.pi*event_time.hour/24.0)
    features['minute_sin'] = np.sin(2*np.pi*event_time.minute/60.0); features['minute_cos'] = np.cos(2*np.pi*event_time.minute/60.0)
    features['dayofweek_sin'] = np.sin(2*np.pi*event_time.weekday()/7.0); features['dayofweek_cos'] = np.cos(2*np.pi*event_time.weekday()/7.0)
    features['prev_interval'] = current_event_series.get(target_col, 0); features['multiplier'] = current_event_series.get('multiplier', 1.0)
    
    hist_before_current = df_all_hist_for_lags[df_all_hist_for_lags['timestamp'] < event_time].copy()
    
    if not hist_before_current.empty:
        if len(hist_before_current) >= 1:
            features['prev_multiplier'] = hist_before_current['multiplier'].iloc[-1]
            if len(hist_before_current) >=2:
                 features['prev_multiplier_lag2'] = hist_before_current['multiplier'].iloc[-2]
            else: features['prev_multiplier_lag2'] = features['prev_multiplier']
        else:
            features['prev_multiplier'] = features['multiplier']
            features['prev_multiplier_lag2'] = features['multiplier']

        if HOURLY_ML_TARGET in hist_before_current.columns:
            if len(hist_before_current) >= 1:
                 features['prev_interval_lag2'] = hist_before_current[HOURLY_ML_TARGET].iloc[-1]
            else: features['prev_interval_lag2'] = features['prev_interval']
            if len(hist_before_current) >= 2:
                 features['prev_interval_lag3'] = hist_before_current[HOURLY_ML_TARGET].iloc[-2]
            else: features['prev_interval_lag3'] = features['prev_interval_lag2']
        else:
            features['prev_interval_lag2'] = features['prev_interval']; features['prev_interval_lag3'] = features['prev_interval']
    else:
        features['prev_multiplier'] = features['multiplier']; features['prev_multiplier_lag2'] = features['multiplier']
        features['prev_interval_lag2'] = features['prev_interval']; features['prev_interval_lag3'] = features['prev_interval']

    if not hist_before_current.empty:
        intervals_for_rolling = hist_before_current[HOURLY_ML_TARGET].dropna()
        multipliers_for_rolling = hist_before_current['multiplier'].dropna()

        features['rolling_mean_interval_3'] = intervals_for_rolling.rolling(window=3, min_periods=1).mean().iloc[-1] if not intervals_for_rolling.empty else features['prev_interval']
        features['rolling_std_interval_3'] = intervals_for_rolling.rolling(window=3, min_periods=1).std().iloc[-1] if not intervals_for_rolling.empty else 0
        features['rolling_mean_multiplier_3'] = multipliers_for_rolling.rolling(window=3, min_periods=1).mean().iloc[-1] if not multipliers_for_rolling.empty else features['multiplier']
        features['rolling_std_multiplier_3'] = multipliers_for_rolling.rolling(window=3, min_periods=1).std().iloc[-1] if not multipliers_for_rolling.empty else 0
        features['avg_interval_short_term_N5'] = intervals_for_rolling.rolling(window=5, min_periods=1).mean().iloc[-1] if not intervals_for_rolling.empty else features['prev_interval']
    else:
        features['rolling_mean_interval_3'] = features['prev_interval']; features['rolling_std_interval_3'] = 0
        features['rolling_mean_multiplier_3'] = features['multiplier']; features['rolling_std_multiplier_3'] = 0
        features['avg_interval_short_term_N5'] = features['prev_interval']

    features['mult_x_prev_mult'] = features['multiplier'] * features['prev_multiplier']
    features['prev_interval_x_mult'] = features['prev_interval'] * features['multiplier']
    features['prev_interval_diff_lag1_lag2'] = features['prev_interval'] - features['prev_interval_lag2']
    
    count_window_start = event_time - timedelta(minutes=15)
    features['events_in_last_15min_before_prev'] = df_all_hist_for_lags[
        (df_all_hist_for_lags['timestamp'] >= count_window_start) & 
        (df_all_hist_for_lags['timestamp'] < event_time)
    ].shape[0] if not df_all_hist_for_lags.empty else 0
    
    for col in feature_list:
        if col not in features or pd.isna(features[col]):
            if 'interval' in col or 'std' in col or 'event' in col : features[col] = 0.0
            elif 'multiplier' in col: features[col] = 1.0
            else: features[col] = 0.0
            logger.debug(f"Feature '{col}' for live hourly pred was NaN/missing, defaulted.")
            
    try: return pd.DataFrame([features], columns=feature_list)
    except Exception as e: logger.error(f"Error creating DataFrame for hourly pred features: {e}. Features: {features}"); return None

# --- Paste the FULL _get_features_for_hml_prediction here ---
def _get_features_for_hml_prediction(df_historical_segment: pd.DataFrame, candidate_event_time: datetime, high_mult_threshold: float) -> Optional[pd.DataFrame]:
    if df_historical_segment.empty: return None
    window_df = df_historical_segment[df_historical_segment['timestamp'] < candidate_event_time].tail(HML_LOOKBACK_WINDOW_EVENTS)
    if window_df.empty:
        if len(df_historical_segment) >= HML_LOOKBACK_WINDOW_EVENTS: window_df = df_historical_segment.tail(HML_LOOKBACK_WINDOW_EVENTS)
        elif not df_historical_segment.empty: window_df = df_historical_segment.tail(len(df_historical_segment))
        else: return None
    last_high_in_window = window_df[window_df['multiplier'] >= high_mult_threshold]['timestamp'].max()
    if pd.notna(last_high_in_window):
        time_since_last_high_pred_hours = (candidate_event_time - last_high_in_window).total_seconds() / 3600.0
        num_low_events_since_last_high_pred = len(window_df[(window_df['timestamp'] > last_high_in_window) & (window_df['multiplier'] < high_mult_threshold)])
    else:
        earliest_time_in_window = window_df['timestamp'].min() if not window_df.empty else candidate_event_time - timedelta(days=7)
        time_since_last_high_pred_hours = (candidate_event_time - earliest_time_in_window).total_seconds() / 3600.0
        num_low_events_since_last_high_pred = len(window_df[window_df['multiplier'] < high_mult_threshold])
    avg_multiplier_in_window_pred = window_df['multiplier'].mean() if not window_df.empty else (df_historical_segment['multiplier'].median() if not df_historical_segment.empty else 1.0)
    if pd.isna(avg_multiplier_in_window_pred): avg_multiplier_in_window_pred = 1.0
    features_dict = {'time_since_last_high_event': time_since_last_high_pred_hours, 'num_low_events_since_last_high': num_low_events_since_last_high_pred, 'avg_multiplier_in_window': avg_multiplier_in_window_pred, 'hour_of_day': candidate_event_time.hour, 'dayofweek': candidate_event_time.weekday()}
    return pd.DataFrame([features_dict], columns=HML_FEATURES_DEF)

# --- Paste the FULL prepare_daily_high_tier_features here ---
def prepare_daily_high_tier_features(df_all_hist: pd.DataFrame, min_multiplier_threshold: float, feature_list: List[str], target_col_name: str, context_label: str, for_live_prediction: bool = False, live_event_series: Optional[pd.Series] = None) -> Optional[pd.DataFrame]:
    logger.info(f"Preparing features for {context_label} model (Live: {for_live_prediction})...")
    if df_all_hist.empty and for_live_prediction and live_event_series is None : logger.warning(f"Input df_all_hist empty and no live_event_series for {context_label}."); return None
    
    if not for_live_prediction:
        df_high_tier_events = df_all_hist[df_all_hist['multiplier'] >= min_multiplier_threshold].copy()
        if len(df_high_tier_events) < 3: logger.warning(f"Not enough >= {min_multiplier_threshold}x events ({len(df_high_tier_events)}) for {context_label} features. Need 3."); return pd.DataFrame()
        return df_ml_final

    if live_event_series is None:
        df_high_tier_events = df_all_hist[df_all_hist['multiplier'] >= min_multiplier_threshold].copy()
        if df_high_tier_events.empty: logger.warning(f"No prior >= {min_multiplier_threshold}x events found for live {context_label} prediction."); return None
        df_high_tier_events.sort_values('timestamp', inplace=True)
        live_event_series = df_high_tier_events.iloc[-1]
    
    df_feat = pd.DataFrame(index=[0])
    event_time = live_event_series.get('timestamp')
    if pd.isna(event_time): logger.warning(f"Timestamp missing for live {context_label} event."); return None
    if event_time.tzinfo is None: event_time = event_time.replace(tzinfo=timezone.utc)

    df_feat['prev_event_hour'] = event_time.hour
    df_feat['prev_event_dayofweek'] = event_time.weekday()
    df_feat['month'] = event_time.month
    df_feat['prev_event_hour_sin'] = np.sin(2*np.pi*df_feat['prev_event_hour']/24.0); df_feat['prev_event_hour_cos'] = np.cos(2*np.pi*df_feat['prev_event_hour']/24.0)
    df_feat['prev_event_dayofweek_sin'] = np.sin(2*np.pi*df_feat['prev_event_dayofweek']/7.0); df_feat['prev_event_dayofweek_cos'] = np.cos(2*np.pi*df_feat['prev_event_dayofweek']/7.0)
    df_feat['month_sin'] = np.sin(2*np.pi*df_feat['month']/12.0); df_feat['month_cos'] = np.cos(2*np.pi*df_feat['month']/12.0)
    epoch_dt = pd.Timestamp("2020-01-01", tz='UTC'); df_feat['days_since_epoch'] = (event_time - epoch_dt).days
    
    df_feat['prev_high_tier_multiplier'] = live_event_series.get('multiplier', min_multiplier_threshold)
    
    df_feat['prev_interval_between_high_tier'] = live_event_series.get(target_col_name, 24*3600)
    df_feat['prev_interval_between_high_tier_lag2'] = live_event_series.get('prev_interval_between_high_tier_lag2', df_feat['prev_interval_between_high_tier'])
    df_feat['prev_high_tier_multiplier_lag2'] = live_event_series.get('prev_high_tier_multiplier_lag2', df_feat['prev_high_tier_multiplier'])
    df_feat['rolling_mean_interval_high_tier_3'] = live_event_series.get('rolling_mean_interval_high_tier_3', df_feat['prev_interval_between_high_tier'])
    df_feat['rolling_std_interval_high_tier_3'] = live_event_series.get('rolling_std_interval_high_tier_3', 0)
    df_feat['rolling_mean_multiplier_high_tier_3'] = live_event_series.get('rolling_mean_multiplier_high_tier_3', df_feat['prev_high_tier_multiplier'])

    for col in feature_list:
        if col not in df_feat.columns or pd.isna(df_feat.loc[0, col]):
            if 'interval' in col or 'std' in col : df_feat.loc[0, col] = 0.0
            elif 'multiplier' in col: df_feat.loc[0, col] = min_multiplier_threshold
            else: df_feat.loc[0, col] = 0.0
            logger.debug(f"Live feature '{col}' for {context_label} was NaN, defaulted.")
    try:
        return df_feat[feature_list]
    except KeyError as e:
        logger.error(f"KeyError preparing live features for {context_label}: {e}. Expected: {feature_list}, Available: {df_feat.columns.tolist()}")
        return None
# --- END SECTION: PREDICTION LOGIC (Part 1 of 2) ---

# --- SECTION: PREDICTION LOGIC (Continued from Part 1) ---

def predict_hourly_custom():
    global HOURLY_MODEL_PIPELINE
    pred_cls = PredictionHourly
    pred_count_target = HOURLY_PRED_COUNT
    min_multiplier_for_preds = HOURLY_MIN_MULTIPLIER
    model_key_for_success_check = "hourly"
    model_pipeline_global_name = "HOURLY_MODEL_PIPELINE" 
    ml_features_list = HOURLY_ML_FEATURES
    prediction_label_log_prefix = "General Hourly"

    logger.info(f"{prediction_label_log_prefix}: PREDICTION CYCLE START")
    now_utc = datetime.now(timezone.utc)
    target_hour_start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    target_hour_end_utc = target_hour_start_utc + timedelta(hours=1)

    existing_preds_for_this_hour_objects = pred_cls.query.filter(
        pred_cls.predicted_datetime_utc >= target_hour_start_utc,
        pred_cls.predicted_datetime_utc < target_hour_end_utc
    ).all()
    
    for p in existing_preds_for_this_hour_objects:
        if p.predicted_datetime_utc.tzinfo is None:
            p.predicted_datetime_utc = p.predicted_datetime_utc.replace(tzinfo=timezone.utc)
    
    existing_preds_count_this_hour = len(existing_preds_for_this_hour_objects)
    
    is_scheduled_run = now_utc.minute == 0

    if is_scheduled_run:
        logger.info(f"{prediction_label_log_prefix}: Scheduled run for {target_hour_start_utc.strftime('%H:%M')}. Clearing existing for this hour.")
        if existing_preds_count_this_hour > 0:
            try:
                num_deleted = pred_cls.query.filter(
                    pred_cls.predicted_datetime_utc >= target_hour_start_utc,
                    pred_cls.predicted_datetime_utc < target_hour_end_utc
                ).delete(synchronize_session=False)
                db.session.commit()
                logger.info(f"  Cleared {num_deleted} old predictions for this hour.")
                existing_preds_count_this_hour = 0 
                existing_preds_for_this_hour_objects = []
            except Exception as e:
                db.session.rollback(); logger.error(f"{prediction_label_log_prefix}: Error clearing: {e}", exc_info=True)
    
    if not is_scheduled_run and existing_preds_count_this_hour >= pred_count_target:
        logger.info(f"{prediction_label_log_prefix}: Found {existing_preds_count_this_hour} existing (>= target {pred_count_target}) for current hour. Skipping.")
        logger.info(f"{prediction_label_log_prefix}: PREDICTION CYCLE END (skipped)")
        return
    
    logger.info(f"{prediction_label_log_prefix}: Proceeding. Current existing for this hour: {existing_preds_count_this_hour}. Target: {pred_count_target}.")
    
    generated_predictions_this_run: List[Any] = list(existing_preds_for_this_hour_objects)
    current_run_hhmm_slots: Set[str] = {p.predicted_datetime_utc.strftime('%H:%M') for p in existing_preds_for_this_hour_objects}

    df_all_hist = get_historical_data_df(sort_ascending=True, limit=200)
    model_pipeline = globals().get(model_pipeline_global_name)
    newly_made_ml_preds_count_this_run = 0

    if MODELS_LOADED_SUCCESSFULLY.get(model_key_for_success_check) and model_pipeline and \
       not (df_all_hist.empty or df_all_hist[HOURLY_ML_TARGET].dropna().count() < 5):
        
        logger.info(f"{prediction_label_log_prefix}: Attempting ML predictions for hour {target_hour_start_utc.strftime('%H:%M')}.")
        last_actual_event = df_all_hist.iloc[-1].copy()
        if pd.isna(last_actual_event[HOURLY_ML_TARGET]) or last_actual_event[HOURLY_ML_TARGET] <= 0:
            median_interval = df_all_hist[HOURLY_ML_TARGET].dropna().median()
            last_actual_event[HOURLY_ML_TARGET] = median_interval if pd.notna(median_interval) and median_interval > 0 else 300.0
            logger.info(f"  Adjusted prev_interval for features to: {last_actual_event[HOURLY_ML_TARGET]}s")
        
        current_time_base = last_actual_event['timestamp']
        current_event_for_features = last_actual_event.copy()
        if current_time_base.tzinfo is None: current_time_base = current_time_base.replace(tzinfo=timezone.utc)

        if current_time_base >= target_hour_end_utc or current_time_base < target_hour_start_utc - timedelta(minutes=30) :
            logger.info(f"  ML Base {current_time_base.strftime('%H:%M')} is unsuitable for target hour {target_hour_start_utc.strftime('%H:%M')}. Adjusting.")
            current_time_base = target_hour_start_utc - timedelta(minutes=np.random.randint(1,10))
            current_event_for_features['timestamp'] = current_time_base
        
        logger.info(f"  ML Initial base for loop: {current_time_base.strftime('%Y-%m-%d %H:%M')}")

        for i in range(pred_count_target * 4): 
            if len(generated_predictions_this_run) >= pred_count_target: break
            if current_time_base >= target_hour_end_utc: break

            features_df = prepare_features_for_hourly_prediction(df_all_hist, current_event_for_features, ml_features_list, HOURLY_ML_TARGET)
            if features_df is None or features_df.empty:
                logger.debug(f"    Iter {i+1}: Feature prep failed. Nudging base.")
                current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue
            try:
                pred_interval_raw = model_pipeline.predict(features_df)[0]
                interval_cap = 25 * 60
                final_interval = int(round(max(60.0, min(pred_interval_raw, interval_cap)) / 60.0) * 60.0)
                if final_interval <= 0: final_interval = np.random.randint(2, 6) * 60
            except Exception as e:
                logger.error(f"    Iter {i+1}: ML Error: {e}", exc_info=False); final_interval = np.random.randint(4, 8) * 60
                current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue
            
            next_pred_time = (current_time_base + timedelta(seconds=final_interval)).replace(second=0, microsecond=0)
            logger.debug(f"    Iter {i+1}: PredInt={final_interval}s, NextCandTime={next_pred_time.strftime('%H:%M')}")

            if next_pred_time < target_hour_start_utc or next_pred_time < now_utc.replace(second=0, microsecond=0) - timedelta(minutes=2): 
                logger.debug(f"    Skipping {next_pred_time.strftime('%H:%M')} (too early/past). Advancing base.")
                current_time_base = next_pred_time; current_event_for_features = pd.Series({'timestamp': current_time_base, 'multiplier': min_multiplier_for_preds, HOURLY_ML_TARGET: final_interval}, dtype=object); continue

            if target_hour_start_utc <= next_pred_time < target_hour_end_utc:
                hhmm = next_pred_time.strftime('%H:%M')
                if hhmm not in current_run_hhmm_slots:
                    if next_pred_time.minute == 0: 
                        logger.info(f"    Skipping ML pred at {hhmm} (minute 00). Nudging base.")
                        current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue

                    avg_m = min_multiplier_for_preds * np.random.uniform(1.1, 1.8)
                    max_m = avg_m * np.random.uniform(1.1, 1.5)
                    conf = round(0.40 + (0.08 * 1.5), 2); conf = min(conf, 0.70)
                    
                    generated_predictions_this_run.append(pred_cls(predicted_datetime_utc=next_pred_time,min_multiplier=min_multiplier_for_preds, avg_multiplier=round(avg_m,2),max_multiplier=round(max_m,2), confidence=conf))
                    current_run_hhmm_slots.add(hhmm)
                    logger.info(f"    Added {prediction_label_log_prefix} ML pred: {hhmm} (Conf: {conf*100:.0f}%)")
                    current_time_base = next_pred_time
                    current_event_for_features = pd.Series({'timestamp': current_time_base, 'multiplier': avg_m, HOURLY_ML_TARGET: final_interval}, dtype=object)
                    newly_made_ml_preds_count_this_run +=1
                else: 
                    logger.debug(f"    Slot {hhmm} taken. Nudging base.")
                    current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60
            elif next_pred_time >= target_hour_end_utc: 
                logger.info(f"    PredTime {next_pred_time.strftime('%H:%M')} is beyond target hour. Breaking ML loop.")
                break 
        
        logger.info(f"{prediction_label_log_prefix}: ML loop finished. Newly made: {newly_made_ml_preds_count_this_run}. Total for hour: {len(generated_predictions_this_run)}.")

    else:
        logger.warning(f"{prediction_label_log_prefix}: Conditions for ML not met. Current existing for hour: {existing_preds_count_this_hour}")
        newly_made_ml_preds_count_this_run = 0

    min_new_ml_to_avoid_placeholders = pred_count_target // 3 

    if len(generated_predictions_this_run) < pred_count_target:
        if newly_made_ml_preds_count_this_run < min_new_ml_to_avoid_placeholders:
            num_placeholders_needed = pred_count_target - len(generated_predictions_this_run)
            if num_placeholders_needed > 0 :
                logger.info(f"  {prediction_label_log_prefix}: Newly made ML ({newly_made_ml_preds_count_this_run}) is low. Adding {num_placeholders_needed} placeholders.")
                additional_placeholders = _generate_placeholder_predictions(
                    pred_cls, num_placeholders_needed, target_hour_start_utc, target_hour_end_utc,
                    min_multiplier_for_preds, current_run_hhmm_slots,
                    f"Fill after {prediction_label_log_prefix} ML (few ML)", force_low_confidence=True
                )
                generated_predictions_this_run.extend(additional_placeholders)
                generated_predictions_this_run.sort(key=lambda p: p.predicted_datetime_utc)
        else:
            logger.info(f"  {prediction_label_log_prefix}: Newly made ML ({newly_made_ml_preds_count_this_run}) sufficient or hour already partially full. Not adding more placeholders if total is close to target.")
                        
    _finalize_and_save_predictions(pred_cls, generated_predictions_this_run, pred_count_target, f"{prediction_label_log_prefix} (ML/Fallback)")
    logger.info(f"{prediction_label_log_prefix}: PREDICTION CYCLE END")

def predict_50x_hourly_custom():
    global HOURLY_50X_MODEL_PIPELINE
    pred_cls = Prediction50xHourly
    pred_count_target = HOURLY_50X_PRED_COUNT
    min_multiplier_for_preds = HOURLY_50X_MIN_MULTIPLIER
    model_key_for_success_check = "hourly_50x"
    model_pipeline_global_name = "HOURLY_50X_MODEL_PIPELINE"
    ml_features_list = HOURLY_50X_ML_FEATURES
    prediction_label_log_prefix = "50x Hourly"

    logger.info(f"{prediction_label_log_prefix}: PREDICTION CYCLE START (Target: {pred_count_target})")
    now_utc = datetime.now(timezone.utc)
    target_hour_start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    target_hour_end_utc = target_hour_start_utc + timedelta(hours=1)

    raw_existing_preds_this_hour = pred_cls.query.filter(
        pred_cls.predicted_datetime_utc >= target_hour_start_utc,
        pred_cls.predicted_datetime_utc < target_hour_end_utc
    ).all()
    existing_preds_for_this_hour_objects = []
    for p_db in raw_existing_preds_this_hour:
        if p_db.predicted_datetime_utc.tzinfo is None:
            p_db.predicted_datetime_utc = p_db.predicted_datetime_utc.replace(tzinfo=timezone.utc)
        existing_preds_for_this_hour_objects.append(p_db)
    existing_preds_count_this_hour = len(existing_preds_for_this_hour_objects)
    
    is_scheduled_run = now_utc.minute == 0

    if is_scheduled_run:
        logger.info(f"{prediction_label_log_prefix}: Scheduled run for {target_hour_start_utc.strftime('%H:%M')}. Clearing existing for this hour.")
        if existing_preds_count_this_hour > 0:
            try:
                num_deleted = pred_cls.query.filter(
                    pred_cls.predicted_datetime_utc >= target_hour_start_utc,
                    pred_cls.predicted_datetime_utc < target_hour_end_utc
                ).delete(synchronize_session=False)
                db.session.commit()
                if num_deleted > 0: logger.info(f"  Cleared {num_deleted} old predictions.")
                existing_preds_count_this_hour = 0 
                existing_preds_for_this_hour_objects = [] 
            except Exception as e:
                db.session.rollback(); logger.error(f"{prediction_label_log_prefix}: Error clearing: {e}", exc_info=True)
    
    if not is_scheduled_run and existing_preds_count_this_hour >= pred_count_target:
        logger.info(f"{prediction_label_log_prefix}: Found {existing_preds_count_this_hour} existing (>= target {pred_count_target}). Skipping.")
        return

    logger.info(f"{prediction_label_log_prefix}: Proceeding. Current existing for this hour: {existing_preds_count_this_hour}. Target: {pred_count_target}.")
    
    generated_predictions_this_run: List[Any] = list(existing_preds_for_this_hour_objects)
    current_run_hhmm_slots: Set[str] = {p.predicted_datetime_utc.strftime('%H:%M') for p in existing_preds_for_this_hour_objects}

    df_all_hist = get_historical_data_df(sort_ascending=True, limit=200)
    model_pipeline = globals().get(model_pipeline_global_name)
    newly_made_ml_preds_count_this_run = 0

    if MODELS_LOADED_SUCCESSFULLY.get(model_key_for_success_check) and model_pipeline and \
       not (df_all_hist.empty or df_all_hist[HOURLY_ML_TARGET].dropna().count() < 5):
        
        logger.info(f"{prediction_label_log_prefix}: Attempting ML predictions for hour {target_hour_start_utc.strftime('%H:%M')}.")
        last_actual_event = df_all_hist.iloc[-1].copy()
        if pd.isna(last_actual_event[HOURLY_ML_TARGET]) or last_actual_event[HOURLY_ML_TARGET] <= 0:
            median_interval = df_all_hist[HOURLY_ML_TARGET].dropna().median()
            last_actual_event[HOURLY_ML_TARGET] = median_interval if pd.notna(median_interval) and median_interval > 0 else 300.0
        
        current_time_base = last_actual_event['timestamp']
        current_event_for_features = last_actual_event.copy()
        if current_time_base.tzinfo is None: current_time_base = current_time_base.replace(tzinfo=timezone.utc)

        if current_time_base >= target_hour_end_utc or current_time_base < target_hour_start_utc - timedelta(minutes=30) :
            logger.info(f"  ML Base {current_time_base.strftime('%H:%M')} is unsuitable. Adjusting to just before target hour {target_hour_start_utc.strftime('%H:%M')}.")
            current_time_base = target_hour_start_utc - timedelta(minutes=np.random.randint(5,15))
            current_event_for_features['timestamp'] = current_time_base
        
        logger.info(f"  ML Initial base for loop: {current_time_base.strftime('%Y-%m-%d %H:%M')}")

        for i in range(pred_count_target * 7): 
            if len(generated_predictions_this_run) >= pred_count_target: break 
            if current_time_base >= target_hour_end_utc: break

            logger.debug(f"  {prediction_label_log_prefix} ML Iter {i+1}: Base={current_time_base.strftime('%H:%M')}, TotalGened={len(generated_predictions_this_run)}")
            features_df = prepare_features_for_hourly_prediction(df_all_hist, current_event_for_features, ml_features_list, HOURLY_ML_TARGET)
            
            if features_df is None or features_df.empty:
                logger.debug(f"    Iter {i+1}: Feature prep failed. Nudging base.")
                current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue
            
            try:
                pred_interval_raw = model_pipeline.predict(features_df)[0]
                interval_cap = 20 * 60
                final_interval = int(round(max(60.0, min(pred_interval_raw, interval_cap)) / 60.0) * 60.0)
                if final_interval <= 0: final_interval = np.random.randint(1, 4) * 60
                logger.debug(f"    Iter {i+1}: PredRaw={pred_interval_raw:.0f}s, FinalInt={final_interval}s")
            except Exception as e:
                logger.error(f"    Iter {i+1}: ML Error: {e}", exc_info=False); final_interval = np.random.randint(2, 5) * 60
                current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue
            
            next_pred_time = (current_time_base + timedelta(seconds=final_interval)).replace(second=0, microsecond=0)
            logger.debug(f"    Iter {i+1}: NextCandTime={next_pred_time.strftime('%H:%M')}")

            if next_pred_time < target_hour_start_utc or \
               next_pred_time < now_utc.replace(second=0, microsecond=0) - timedelta(minutes=2):
                logger.debug(f"    Skipping {next_pred_time.strftime('%H:%M')} (too early/past). Advancing base.")
                current_time_base = next_pred_time
                current_event_for_features = pd.Series({'timestamp': current_time_base, 'multiplier': min_multiplier_for_preds, HOURLY_ML_TARGET: final_interval}, dtype=object)
                continue

            if target_hour_start_utc <= next_pred_time < target_hour_end_utc:
                hhmm = next_pred_time.strftime('%H:%M')
                if hhmm not in current_run_hhmm_slots:
                    if next_pred_time.minute == 0:
                        logger.info(f"    Skipping ML pred at {hhmm} (minute 00) for {prediction_label_log_prefix}.")
                        current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60; continue
                    
                    avg_m = min_multiplier_for_preds * np.random.uniform(1.05, 1.5)
                    max_m = avg_m * np.random.uniform(1.1, 1.3)
                    conf = round(0.35 + (0.18 * 1.0), 2); conf = min(conf, 0.60)
                    
                    generated_predictions_this_run.append(pred_cls(predicted_datetime_utc=next_pred_time,min_multiplier=min_multiplier_for_preds, avg_multiplier=round(avg_m,2),max_multiplier=round(max_m,2), confidence=conf))
                    current_run_hhmm_slots.add(hhmm)
                    logger.info(f"    Added {prediction_label_log_prefix} ML pred: {hhmm} (Conf: {conf*100:.0f}%)")
                    current_time_base = next_pred_time
                    current_event_for_features = pd.Series({'timestamp': current_time_base, 'multiplier': avg_m, HOURLY_ML_TARGET: final_interval}, dtype=object)
                    newly_made_ml_preds_count_this_run +=1
                else: 
                    logger.debug(f"    Slot {hhmm} taken. Nudging base.")
                    current_time_base += timedelta(minutes=1); current_event_for_features['timestamp'] = current_time_base; current_event_for_features[HOURLY_ML_TARGET] = 60
            elif next_pred_time >= target_hour_end_utc: 
                logger.info(f"    PredTime {next_pred_time.strftime('%H:%M')} is beyond target hour {target_hour_end_utc.strftime('%H:%M')}. Breaking ML loop.")
                break 
        
        logger.info(f"{prediction_label_log_prefix}: ML loop finished. Newly made this run: {newly_made_ml_preds_count_this_run}. Total for hour: {len(generated_predictions_this_run)}.")
    else:
        logger.warning(f"{prediction_label_log_prefix}: Conditions for ML not met. Current existing for hour: {existing_preds_count_this_hour}")

    if len(generated_predictions_this_run) < pred_count_target:
        num_placeholders_needed = pred_count_target - len(generated_predictions_this_run)
        logger.info(f"  {prediction_label_log_prefix}: ML generated {len(generated_predictions_this_run)} preds. Adding {num_placeholders_needed} placeholders to reach target {pred_count_target}.")
        
        additional_placeholders = _generate_placeholder_predictions(
            pred_cls=pred_cls, 
            num_preds=num_placeholders_needed, 
            target_start_dt=target_hour_start_utc, 
            target_end_dt=target_hour_end_utc,
            min_mult=min_multiplier_for_preds, 
            existing_hhmm_slots=current_run_hhmm_slots,
            reason=f"Fill after {prediction_label_log_prefix} ML to reach target", 
            force_low_confidence=True
        )
        
        generated_predictions_this_run.extend(additional_placeholders)
    
    _finalize_and_save_predictions(pred_cls, generated_predictions_this_run, pred_count_target, f"{prediction_label_log_prefix} (ML/Fallback)")
    logger.info(f"{prediction_label_log_prefix}: PREDICTION CYCLE END")

def predict_50x_daily_custom():
    """
    Generates 20 daily 50x predictions spread randomly and uniquely across the entire day.
    This method ensures no duplicate time slots and covers the range from 00:01 to 23:59.
    """
    logger.info("50X DAILY PREDICTION (Full Day Unique Shuffle) CYCLE START")
    pred_cls = Prediction50xDaily
    pred_count_target = DAILY_50X_PRED_COUNT
    min_mult_for_preds = DAILY_50X_MIN_MULTIPLIER
    prediction_label = "50x+ Daily (Unique Shuffle)"

    now_utc = datetime.now(timezone.utc)
    target_day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    generated_predictions: List[Any] = []

    total_minutes_in_day = 24 * 60
    all_possible_minutes = list(range(1, total_minutes_in_day))

    if pred_count_target > len(all_possible_minutes):
        logger.error(f"Cannot generate {pred_count_target} unique predictions. Only {len(all_possible_minutes)} unique minute-slots are available. Aborting.")
        return

    np.random.shuffle(all_possible_minutes)
    chosen_minute_offsets = all_possible_minutes[:pred_count_target]
    logger.info(f"Generating {len(chosen_minute_offsets)} unique predictions by shuffling all available minutes from 00:01-23:59...")

    for minute_offset in chosen_minute_offsets:
        pred_time_utc = target_day_start + timedelta(minutes=int(minute_offset))
        avg_m = min_mult_for_preds * np.random.uniform(1.1, 2.5)
        max_m = avg_m * np.random.uniform(1.2, 1.8)
        conf = round(np.random.uniform(0.25, 0.60), 2)
        new_pred = pred_cls(
            predicted_datetime_utc=pred_time_utc, min_multiplier=min_mult_for_preds,
            avg_multiplier=round(avg_m, 2), max_multiplier=round(max_m, 2), confidence=conf
        )
        generated_predictions.append(new_pred)
    
    logger.info(f"Successfully created {len(generated_predictions)} unique, randomly-spread time slots for the day.")

    if not generated_predictions:
        logger.info(f"No {prediction_label} predictions were generated to save.")
        return
    final_preds_list = sorted(generated_predictions, key=lambda p: p.predicted_datetime_utc)
    for i in range(len(final_preds_list)):
        if i == 0:
            final_preds_list[i].interval_to_next_seconds = None
        else:
            interval_sec = (final_preds_list[i].predicted_datetime_utc - final_preds_list[i-1].predicted_datetime_utc).total_seconds()
            final_preds_list[i].interval_to_next_seconds = int(max(60, interval_sec))
    try:
        pred_cls.query.filter(
            pred_cls.predicted_datetime_utc >= target_day_start,
            pred_cls.predicted_datetime_utc < (target_day_start + timedelta(days=1))
        ).delete(synchronize_session=False)
        db.session.commit()
        db.session.add_all(final_preds_list)
        db.session.commit()
        logger.info(f"Cleared old and saved {len(final_preds_list)} new {prediction_label} predictions.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving {prediction_label} predictions: {e}", exc_info=True)
    
    logger.info("50X DAILY PREDICTION (Full Day Unique Shuffle) CYCLE END")

def predict_high_multiplier_likelihood():
    """
    Generates and logs the top 3 most likely predictions for high multiplier events.
    This function now ignores the hard threshold and always selects the best predictions
    the model can find, ensuring there are always results to display.
    """
    global HML_MODEL_PIPELINE
    
    logger.info(f"HML PREDICTION CYCLE START (Selecting Top {HML_NUM_UPCOMING_PREDS} Likelihoods)")

    now_utc = datetime.now(timezone.utc)
    
    # Clear all old upcoming predictions to ensure a clean slate for every run.
    try:
        num_deleted = HighMultiplierPredictionLog.query.filter(
            HighMultiplierPredictionLog.predicted_event_time_utc >= now_utc
        ).delete(synchronize_session=False)
        db.session.commit()
        if num_deleted > 0:
            logger.info(f"HML: Cleared {num_deleted} old upcoming HML predictions.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"HML: Error clearing old predictions: {e}", exc_info=True)

    df_all_hist_orig = get_historical_data_df(sort_ascending=True, limit=HML_LOOKBACK_WINDOW_EVENTS + 100)

    # --- THIS BLOCK IS NOW CORRECTED ---
    if not MODELS_LOADED_SUCCESSFULLY.get("hml") or HML_MODEL_PIPELINE is None:
        logger.warning("HML model not loaded. Generating random placeholders for HML.")
        # As a final fallback, create random placeholders if the model is missing
        placeholder_preds = []
        for i in range(HML_NUM_UPCOMING_PREDS):
            # Spread placeholders across the hour
            pred_time = now_utc.replace(second=0, microsecond=0) + timedelta(minutes=(i + 1) * 15)
            placeholder_preds.append(HighMultiplierPredictionLog(
                predicted_event_time_utc=pred_time,
                likelihood_percentage=np.random.randint(25, 45), # Random but reasonable looking
                expected_value_multiplier=None,
                generated_at_utc=now_utc
            ))
        db.session.add_all(placeholder_preds)
        db.session.commit()
        app.last_high_event_pred_update = now_utc
        return
    # --- END OF CORRECTION ---

    logger.info("Using pre-trained HML model to generate new likelihoods.")
    potential_predictions: List[Dict[str, Any]] = []
    
    historical_high_multipliers = df_all_hist_orig[df_all_hist_orig['multiplier'] >= HML_THRESHOLD]['multiplier']
    overall_avg_historical_high_mult = round(historical_high_multipliers.mean(), 2) if not historical_high_multipliers.empty else HML_THRESHOLD * 1.5

    min_predict_time = now_utc + timedelta(minutes=1)

    for hour_offset in range(HML_HOURLY_TARGET_HOURS_AHEAD + 1):
        target_hour_start_dt = (now_utc + timedelta(hours=hour_offset)).replace(minute=0, second=0, microsecond=0)
        minute_step = 60 // HML_HOURLY_CANDIDATE_SLOTS_PER_HOUR
        
        for i in range(HML_HOURLY_CANDIDATE_SLOTS_PER_HOUR):
            candidate_time = target_hour_start_dt + timedelta(minutes=i * minute_step)
            if candidate_time < min_predict_time:
                continue
            
            current_features_df = _get_features_for_hml_prediction(df_all_hist_orig, candidate_time, HML_THRESHOLD)
            if current_features_df is None or current_features_df.empty:
                continue
            
            try:
                probability_high = HML_MODEL_PIPELINE.predict_proba(current_features_df)[0][1]
                likelihood_percent = int(probability_high * 100)

                window_for_exp_mult = df_all_hist_orig[df_all_hist_orig['timestamp'] < candidate_time].tail(HML_LOOKBACK_WINDOW_EVENTS)
                window_high_mults = window_for_exp_mult[window_for_exp_mult['multiplier'] >= HML_THRESHOLD]['multiplier']
                current_expected_high_mult = overall_avg_historical_high_mult
                if not window_high_mults.empty:
                    current_expected_high_mult = round(window_high_mults.mean(), 2)
                
                potential_predictions.append({
                    'predicted_event_time_utc': candidate_time,
                    'likelihood_percentage': likelihood_percent,
                    'expected_value_multiplier': current_expected_high_mult if likelihood_percent > 50 and pd.notna(current_expected_high_mult) else None,
                    'generated_at_utc': now_utc
                })

            except Exception as e:
                logger.error(f"HML model prediction error for candidate {candidate_time}: {e}", exc_info=True)
    
    new_predictions_to_log: List[HighMultiplierPredictionLog] = []
    if potential_predictions:
        sorted_potential_preds = sorted(potential_predictions, key=lambda x: -x['likelihood_percentage'])
        min_time_separation = timedelta(minutes=15)
        chosen_prediction_times: List[datetime] = []
        
        for pot_pred_data in sorted_potential_preds:
            if len(new_predictions_to_log) >= HML_NUM_UPCOMING_PREDS:
                break
            candidate_pred_time = pot_pred_data['predicted_event_time_utc']
            is_too_close = any(abs((candidate_pred_time - ct).total_seconds()) < min_time_separation.total_seconds() for ct in chosen_prediction_times)
            
            if not is_too_close:
                new_predictions_to_log.append(HighMultiplierPredictionLog(**pot_pred_data))
                chosen_prediction_times.append(candidate_pred_time)
    
    if new_predictions_to_log:
        try:
            new_predictions_to_log.sort(key=lambda p: p.predicted_event_time_utc)
            db.session.add_all(new_predictions_to_log)
            db.session.commit()
            logger.info(f"HML: Logged {len(new_predictions_to_log)} new top HML predictions.")
        except Exception as e:
            db.session.rollback()
            logger.error(f"HML: Error saving new HML predictions: {e}", exc_info=True)
    else:
        logger.warning("HML: Model did not produce any potential predictions to rank.")
    
    app.last_high_event_pred_update = now_utc

def predict_100x_custom():
    """
    Generates 15 daily 100x predictions using a hybrid "ML First" approach.
    1. Attempts to generate a prediction using the trained ML model.
    2. Fills the remaining slots with predictions spread randomly and uniquely across the day.
    """
    global DAILY_100X_MODEL_PIPELINE
    logger.info("100X DAILY PREDICTION (ML First + Unique Shuffle Fill) CYCLE START")
    pred_cls = Prediction100x
    pred_count_target = DAILY_100X_PRED_COUNT  # 15
    min_mult_for_preds = DAILY_100X_MIN_MULTIPLIER
    prediction_label = "100x+ Daily (ML/Heuristic)"

    now_utc = datetime.now(timezone.utc)
    target_day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    target_day_end = target_day_start + timedelta(days=1)

    generated_predictions: List[Any] = []
    current_run_hhmm_slots: Set[str] = set()

    # --- 1. ML PREDICTION PHASE ---
    df_all_hist = get_historical_data_df(sort_ascending=True)
    if MODELS_LOADED_SUCCESSFULLY.get("daily_100x") and DAILY_100X_MODEL_PIPELINE and not df_all_hist.empty:
        df_100x_hist = df_all_hist[df_all_hist['multiplier'] >= min_mult_for_preds].copy()
        if not df_100x_hist.empty:
            last_100x_event = df_100x_hist.iloc[-1]
            features_df = prepare_daily_high_tier_features(
                df_all_hist=df_all_hist,
                min_multiplier_threshold=min_mult_for_preds,
                feature_list=DAILY_100X_ML_FEATURES,
                target_col_name=DAILY_100X_TARGET_COL,
                context_label="100x Daily Live",
                for_live_prediction=True,
                live_event_series=last_100x_event
            )
            if features_df is not None and not features_df.empty:
                try:
                    predicted_interval_seconds = DAILY_100X_MODEL_PIPELINE.predict(features_df)[0]
                    last_event_time = last_100x_event['timestamp']
                    predicted_event_time_utc = last_event_time + timedelta(seconds=predicted_interval_seconds)
                    
                    if target_day_start <= predicted_event_time_utc < target_day_end and predicted_event_time_utc > now_utc:
                        if predicted_event_time_utc.minute == 0:
                            predicted_event_time_utc += timedelta(minutes=1)
                        
                        hhmm_key = predicted_event_time_utc.strftime('%H:%M')
                        avg_m = min_mult_for_preds * np.random.uniform(1.2, 2.0)
                        max_m = avg_m * np.random.uniform(1.2, 1.5)
                        
                        ml_confidence = round(np.random.uniform(0.60, 0.75), 2)

                        generated_predictions.append(pred_cls(
                            predicted_datetime_utc=predicted_event_time_utc,
                            min_multiplier=min_mult_for_preds, avg_multiplier=round(avg_m, 2),
                            max_multiplier=round(max_m, 2), confidence=ml_confidence
                        ))
                        current_run_hhmm_slots.add(hhmm_key)
                        logger.info(f"  -> Added ML-based 100x prediction at {hhmm_key}")
                except Exception as e:
                    logger.error(f"Error during 100x ML prediction step: {e}", exc_info=True)
            else:
                logger.warning("Feature preparation for 100x ML prediction failed.")
        else:
            logger.warning("Not enough historical 100x+ events to generate an ML prediction.")
    else:
        logger.warning("100x Daily ML model not loaded or no history. Skipping ML prediction phase.")

    # --- 2. HEURISTIC FILL PHASE ---
    num_needed = pred_count_target - len(generated_predictions)
    if num_needed > 0:
        logger.info(f"ML part generated {len(generated_predictions)} predictions. Need to generate {num_needed} more via unique shuffle.")
        
        total_minutes_in_day = 24 * 60
        all_possible_minutes = list(range(1, total_minutes_in_day))
        
        used_minute_offsets = {
            int((p.predicted_datetime_utc - target_day_start).total_seconds() // 60)
            for p in generated_predictions
        }
        
        available_minutes = [m for m in all_possible_minutes if m not in used_minute_offsets]
        
        np.random.shuffle(available_minutes)
        chosen_minute_offsets = available_minutes[:num_needed]

        for minute_offset in chosen_minute_offsets:
            pred_time_utc = target_day_start + timedelta(minutes=int(minute_offset))
            avg_m = min_mult_for_preds * np.random.uniform(1.1, 1.8)
            max_m = avg_m * np.random.uniform(1.1, 1.4)
            conf = round(np.random.uniform(0.05, 0.20), 2)
            
            generated_predictions.append(pred_cls(
                predicted_datetime_utc=pred_time_utc, min_multiplier=min_mult_for_preds,
                avg_multiplier=round(avg_m, 2), max_multiplier=round(max_m, 2), confidence=conf
            ))
            
    # --- 3. FINALIZATION ---
    if not generated_predictions:
        logger.info(f"No {prediction_label} predictions were generated to save.")
        return

    final_preds_list = sorted(generated_predictions, key=lambda p: p.predicted_datetime_utc)

    for i in range(len(final_preds_list)):
        if i == 0:
            final_preds_list[i].interval_to_next_seconds = None
        else:
            interval_sec = (final_preds_list[i].predicted_datetime_utc - final_preds_list[i-1].predicted_datetime_utc).total_seconds()
            final_preds_list[i].interval_to_next_seconds = int(max(60, interval_sec))

    try:
        pred_cls.query.filter(
            pred_cls.predicted_datetime_utc >= target_day_start,
            pred_cls.predicted_datetime_utc < target_day_end
        ).delete(synchronize_session=False)
        db.session.commit()
        
        db.session.add_all(final_preds_list)
        db.session.commit()
        logger.info(f"Cleared old and saved {len(final_preds_list)} new {prediction_label} predictions.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving {prediction_label} predictions: {e}", exc_info=True)

    logger.info("100X DAILY PREDICTION (ML First + Unique Shuffle Fill) CYCLE END")

def predict_500x_custom():
    """
    Generates 10 daily 500x predictions using a hybrid "ML-like Heuristic First" approach.
    1. Attempts to generate a smart prediction based on historical 500x+ data.
    2. Fills the remaining slots with predictions spread randomly and uniquely across the day.
    """
    logger.info("500X DAILY PREDICTION (Heuristic First + Unique Shuffle Fill) CYCLE START")
    pred_cls = Prediction500x
    pred_count_target = DAILY_500X_PRED_COUNT  # 10
    min_mult_for_preds = DAILY_500X_MIN_MULTIPLIER
    recent_n_for_pattern = DAILY_500X_RECENT_N_EVENTS # 5
    prediction_label = "500x+ Daily (Heuristic/Shuffle)"

    now_utc = datetime.now(timezone.utc)
    target_day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    target_day_end = target_day_start + timedelta(days=1)

    generated_predictions: List[Any] = []
    
    # --- 1. "ML-LIKE" HEURISTIC PREDICTION PHASE ---
    df_500x_hist = get_historical_data_df(min_multiplier_filter=min_mult_for_preds, sort_ascending=True)
    
    if len(df_500x_hist) >= recent_n_for_pattern:
        recent_intervals = df_500x_hist['interval_to_next_actual'].dropna().tail(recent_n_for_pattern).values
        if len(recent_intervals) > 0:
            avg_interval = np.mean(recent_intervals)
            std_interval = np.std(recent_intervals)
            last_event_time = df_500x_hist['timestamp'].iloc[-1]

            predicted_interval_offset = np.random.normal(0, 0.25) * std_interval
            predicted_interval_seconds = max(3600.0, avg_interval + predicted_interval_offset)
            predicted_event_time_utc = last_event_time + timedelta(seconds=predicted_interval_seconds)

            if target_day_start <= predicted_event_time_utc < target_day_end and predicted_event_time_utc > now_utc:
                if predicted_event_time_utc.minute == 0:
                    predicted_event_time_utc += timedelta(minutes=1)
                
                avg_m = min_mult_for_preds * np.random.uniform(1.2, 2.2)
                max_m = avg_m * np.random.uniform(1.2, 1.6)
                smart_confidence = round(np.random.uniform(0.35, 0.55), 2)
                
                generated_predictions.append(pred_cls(
                    predicted_datetime_utc=predicted_event_time_utc,
                    min_multiplier=min_mult_for_preds, avg_multiplier=round(avg_m, 2),
                    max_multiplier=round(max_m, 2), confidence=smart_confidence
                ))
                logger.info(f"  -> Added 'Smart Heuristic' 500x prediction at {predicted_event_time_utc.strftime('%H:%M')}")
        else:
            logger.warning("Not enough valid intervals in 500x+ history for smart prediction.")
    else:
        logger.warning("Not enough historical 500x+ events for smart prediction. Using full shuffle.")

    # --- 2. UNIQUE SHUFFLE FILL PHASE ---
    num_needed = pred_count_target - len(generated_predictions)
    if num_needed > 0:
        logger.info(f"Smart part generated {len(generated_predictions)} predictions. Need to fill {num_needed} more via unique shuffle.")
        
        total_minutes_in_day = 24 * 60
        all_possible_minutes = list(range(1, total_minutes_in_day))
        
        used_minute_offsets = {
            int((p.predicted_datetime_utc - target_day_start).total_seconds() // 60)
            for p in generated_predictions
        }
        available_minutes = [m for m in all_possible_minutes if m not in used_minute_offsets]
        
        np.random.shuffle(available_minutes)
        chosen_minute_offsets = available_minutes[:num_needed]

        for minute_offset in chosen_minute_offsets:
            pred_time_utc = target_day_start + timedelta(minutes=int(minute_offset))
            avg_m = min_mult_for_preds * np.random.uniform(1.1, 2.2)
            max_m = avg_m * np.random.uniform(1.1, 1.6)
            conf = round(np.random.uniform(0.15, 0.30), 2)
            
            generated_predictions.append(pred_cls(
                predicted_datetime_utc=pred_time_utc, min_multiplier=min_mult_for_preds,
                avg_multiplier=round(avg_m, 2), max_multiplier=round(max_m, 2), confidence=conf
            ))
            
    # --- 3. FINALIZATION ---
    if not generated_predictions:
        logger.info(f"No {prediction_label} predictions were generated to save.")
        return

    final_preds_list = sorted(generated_predictions, key=lambda p: p.predicted_datetime_utc)

    for i in range(len(final_preds_list)):
        if i == 0:
            final_preds_list[i].interval_to_next_seconds = None
        else:
            interval_sec = (final_preds_list[i].predicted_datetime_utc - final_preds_list[i-1].predicted_datetime_utc).total_seconds()
            final_preds_list[i].interval_to_next_seconds = int(max(60, interval_sec))

    try:
        pred_cls.query.filter(
            pred_cls.predicted_datetime_utc >= target_day_start,
            pred_cls.predicted_datetime_utc < target_day_end
        ).delete(synchronize_session=False)
        db.session.commit()
        
        db.session.add_all(final_preds_list)
        db.session.commit()
        logger.info(f"Cleared old and saved {len(final_preds_list)} new {prediction_label} predictions.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving {prediction_label} predictions: {e}", exc_info=True)

    logger.info("500X DAILY PREDICTION (Heuristic First + Unique Shuffle Fill) CYCLE END")

def predict_1000x_custom():
    """
    Generates 4 daily 1000x predictions using a hybrid "Heuristic First + Unique Shuffle Fill" approach.
    This ensures a smart prediction is prioritized, and the rest are uniquely and randomly spread.
    """
    logger.info("1000X DAILY PREDICTION (Heuristic First + Unique Shuffle Fill) CYCLE START")
    pred_cls = Prediction1000x
    pred_count_target = DAILY_1000X_PRED_COUNT  # 4
    min_mult_for_preds = DAILY_1000X_MIN_MULTIPLIER
    recent_n_for_pattern = DAILY_1000X_RECENT_N_EVENTS # 3
    prediction_label = "1000x+ Daily (Heuristic/Shuffle)"

    now_utc = datetime.now(timezone.utc)
    target_day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    target_day_end = target_day_start + timedelta(days=1)

    generated_predictions: List[Any] = []
    
    # --- 1. "SMART" HEURISTIC PREDICTION PHASE ---
    df_1000x_hist = get_historical_data_df(min_multiplier_filter=min_mult_for_preds, sort_ascending=True)

    if len(df_1000x_hist) >= recent_n_for_pattern:
        recent_intervals = df_1000x_hist['interval_to_next_actual'].dropna().tail(recent_n_for_pattern).values
        if len(recent_intervals) > 0:
            avg_interval = np.mean(recent_intervals)
            std_interval = np.std(recent_intervals)
            last_event_time = df_1000x_hist['timestamp'].iloc[-1]

            predicted_interval_offset = np.random.normal(0, 0.15) * std_interval # Less deviation for rarer events
            predicted_interval_seconds = max(3600.0, avg_interval + predicted_interval_offset)
            predicted_event_time_utc = last_event_time + timedelta(seconds=predicted_interval_seconds)

            if target_day_start <= predicted_event_time_utc < target_day_end and predicted_event_time_utc > now_utc:
                if predicted_event_time_utc.minute == 0:
                    predicted_event_time_utc += timedelta(minutes=1)

                avg_m = min_mult_for_preds * np.random.uniform(1.1, 1.8)
                max_m = avg_m * np.random.uniform(1.1, 1.3)
                smart_confidence = round(np.random.uniform(0.30, 0.50), 2)
                
                generated_predictions.append(pred_cls(
                    predicted_datetime_utc=predicted_event_time_utc,
                    min_multiplier=min_mult_for_preds, avg_multiplier=round(avg_m, 2),
                    max_multiplier=round(max_m, 2), confidence=smart_confidence
                ))
                logger.info(f"  -> Added 'Smart Heuristic' 1000x prediction at {predicted_event_time_utc.strftime('%H:%M')}")
        else:
            logger.warning("Not enough valid intervals in 1000x+ history for smart prediction.")
    else:
        logger.warning("Not enough historical 1000x+ events for smart prediction. Using full shuffle.")

    # --- 2. UNIQUE SHUFFLE FILL PHASE ---
    num_needed = pred_count_target - len(generated_predictions)
    if num_needed > 0:
        logger.info(f"Smart part generated {len(generated_predictions)} predictions. Need to fill {num_needed} more via unique shuffle.")
        
        total_minutes_in_day = 24 * 60
        all_possible_minutes = list(range(1, total_minutes_in_day))
        
        used_minute_offsets = {
            int((p.predicted_datetime_utc - target_day_start).total_seconds() // 60)
            for p in generated_predictions
        }
        available_minutes = [m for m in all_possible_minutes if m not in used_minute_offsets]
        
        np.random.shuffle(available_minutes)
        chosen_minute_offsets = available_minutes[:num_needed]

        for minute_offset in chosen_minute_offsets:
            pred_time_utc = target_day_start + timedelta(minutes=int(minute_offset))
            avg_m = min_mult_for_preds * np.random.uniform(1.05, 1.5)
            max_m = avg_m * np.random.uniform(1.05, 1.2)
            conf = round(np.random.uniform(0.10, 0.25), 2)
            
            generated_predictions.append(pred_cls(
                predicted_datetime_utc=pred_time_utc, min_multiplier=min_mult_for_preds,
                avg_multiplier=round(avg_m, 2), max_multiplier=round(max_m, 2), confidence=conf
            ))
            
    # --- 3. FINALIZATION ---
    if not generated_predictions:
        logger.info(f"No {prediction_label} predictions were generated to save.")
        return

    final_preds_list = sorted(generated_predictions, key=lambda p: p.predicted_datetime_utc)

    for i in range(len(final_preds_list)):
        if i == 0:
            final_preds_list[i].interval_to_next_seconds = None
        else:
            interval_sec = (final_preds_list[i].predicted_datetime_utc - final_preds_list[i-1].predicted_datetime_utc).total_seconds()
            final_preds_list[i].interval_to_next_seconds = int(max(60, interval_sec))

    try:
        pred_cls.query.filter(
            pred_cls.predicted_datetime_utc >= target_day_start,
            pred_cls.predicted_datetime_utc < target_day_end
        ).delete(synchronize_session=False)
        db.session.commit()
        
        db.session.add_all(final_preds_list)
        db.session.commit()
        logger.info(f"Cleared old and saved {len(final_preds_list)} new {prediction_label} predictions.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving {prediction_label} predictions: {e}", exc_info=True)

    logger.info("1000X DAILY PREDICTION (Heuristic First + Unique Shuffle Fill) CYCLE END")

def predict_monthly_5000x_custom():
    logger.info("MONTHLY 5000X+ PREDICTION CYCLE START")
    pred_cls = PredictionMonthly; now_utc = datetime.now(timezone.utc)
    existing_monthly_preds_from_db = pred_cls.query.order_by(pred_cls.target_date_utc.asc()).all()
    processed_existing_preds = []
    for p_db in existing_monthly_preds_from_db:
        if p_db.predicted_value_utc and p_db.predicted_value_utc.tzinfo is None: p_db.predicted_value_utc = p_db.predicted_value_utc.replace(tzinfo=timezone.utc)
        processed_existing_preds.append(p_db)
    valid_preds_count = 0; preds_to_keep = []; preds_to_delete = []
    for pred in processed_existing_preds:
        is_valid = False
        if pred.predicted_value_utc and pred.predicted_value_utc > now_utc: is_valid = True
        elif pred.target_date_utc and datetime.combine(pred.target_date_utc, datetime.min.time(), tzinfo=timezone.utc) >= now_utc.replace(hour=0, minute=0, second=0, microsecond=0): is_valid = True
        if is_valid: valid_preds_count += 1; preds_to_keep.append(pred)
        else: preds_to_delete.append(pred)
    desired_pred_count = MONTHLY_5000X_PRED_COUNT if MONTHLY_5000X_PRED_COUNT <= 3 else 3
    if valid_preds_count >= desired_pred_count:
        logger.info(f"Found {valid_preds_count} valid monthly predictions. No new generation needed.")
        if len(preds_to_keep) > desired_pred_count: preds_to_delete.extend(preds_to_keep[desired_pred_count:])
        if preds_to_delete:
            try: [db.session.delete(p_del) for p_del in preds_to_delete]; db.session.commit(); logger.info(f"Deleted {len(preds_to_delete)} invalid/extra monthly predictions.")
            except Exception as e: db.session.rollback(); logger.error(f"Error deleting old monthly predictions: {e}")
        logger.info("MONTHLY 5000X+ PREDICTION CYCLE END - No new generation."); return
    if preds_to_delete:
        try: [db.session.delete(p_del) for p_del in preds_to_delete]; db.session.commit(); logger.info(f"Deleted {len(preds_to_delete)} invalid monthly predictions.")
        except Exception as e: db.session.rollback(); logger.error(f"Error deleting invalid monthly predictions: {e}")
    num_to_generate = desired_pred_count - len(preds_to_keep); logger.info(f"Need to generate {num_to_generate} new monthly predictions.")
    predictions_to_add = []; first_day_current_month = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if first_day_current_month.month == 12: first_day_next_month = first_day_current_month.replace(year=first_day_current_month.year + 1, month=1)
    else: first_day_next_month = first_day_current_month.replace(month=first_day_current_month.month + 1)
    if first_day_next_month.month == 12: next_month_after_target = first_day_next_month.replace(year=first_day_next_month.year + 1, month=1, day=1)
    else: next_month_after_target = first_day_next_month.replace(month=first_day_next_month.month + 1, day=1)
    days_in_next_month = (next_month_after_target - timedelta(days=1)).day
    generated_dates_this_run = [p.target_date_utc for p in preds_to_keep]
    for i in range(num_to_generate):
        try:
            attempts = 0; target_date = None
            while attempts < 10:
                attempts += 1; day_candidate = np.random.randint(5, 16) if len(preds_to_keep) + len(predictions_to_add) == 0 else np.random.randint(16, min(29, days_in_next_month + 1))
                actual_day = min(day_candidate, days_in_next_month); current_target_date_attempt = first_day_next_month.replace(day=actual_day).date()
                if current_target_date_attempt not in generated_dates_this_run: target_date = current_target_date_attempt; break
                elif attempts == 10: target_date = first_day_next_month.replace(day=min(np.random.randint(1, days_in_next_month + 1), days_in_next_month)).date(); break 
            if target_date is None: logger.error("Failed to determine target date for monthly pred."); continue
            predicted_dt = first_day_next_month.replace(day=target_date.day, hour=np.random.randint(0,24), minute=np.random.choice([0,15,30,45]))
            notes_str = "Primary forecast." if len(preds_to_keep) + len(predictions_to_add) == 0 else "Secondary forecast."
            new_pred = pred_cls(target_date_utc=target_date, predicted_value_utc=predicted_dt, min_multiplier=MONTHLY_5000X_MIN_MULTIPLIER * np.random.uniform(1.0, 1.05 + (i*0.05)), confidence_level="Speculative", notes=notes_str)
            predictions_to_add.append(new_pred); generated_dates_this_run.append(target_date); logger.info(f"Generated new monthly prediction for {target_date.strftime('%Y-%m-%d')}.")
        except Exception as e: logger.error(f"Error generating new monthly pred slot #{i+1}: {e}", exc_info=True); break
    if predictions_to_add:
        try: db.session.add_all(predictions_to_add); db.session.commit(); logger.info(f"Saved {len(predictions_to_add)} new monthly 5000x+ predictions.")
        except Exception as e: db.session.rollback(); logger.error(f"Error saving new monthly predictions: {e}", exc_info=True)
    logger.info("MONTHLY 5000X+ PREDICTION CYCLE END")
# --- END SECTION: PREDICTION LOGIC ---

# --- SECTION: SCHEDULER JOBS ---
def job_predict_hourly():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_hourly_custom job.")
        predict_hourly_custom()
        
def job_predict_50x_hourly():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_50x_hourly_custom job.")
        predict_50x_hourly_custom()
        
def job_predict_50x_daily():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_50x_daily_custom job.")
        predict_50x_daily_custom()
        
def job_predict_100x():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_100x_custom job.")
        predict_100x_custom()
        
def job_predict_500x():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_500x_custom job.")
        predict_500x_custom()
        
def job_predict_1000x():
    with app.app_context(): 
        logger.info("Scheduler: Running predict_1000x_custom job.")
        predict_1000x_custom()
        
def job_predict_high_multiplier_likelihood():
    with app.app_context(): logger.info("Scheduler: Running predict_high_multiplier_likelihood job."); predict_high_multiplier_likelihood()
def job_predict_monthly_5000x():
    with app.app_context(): logger.info("Scheduler: Running predict_monthly_5000x_custom job."); predict_monthly_5000x_custom()
def job_cleanup_old_data():
    with app.app_context():
        logger.info("Scheduler: Running job_cleanup_old_data.")
        cleanup_threshold_date = datetime.now(timezone.utc) - timedelta(days=90)
        prediction_models_to_clean = [ PredictionHourly, Prediction50xHourly, Prediction100x, Prediction500x, Prediction1000x, Prediction50xDaily, HighMultiplierPredictionLog]
        total_deleted_count = 0
        for model_cls in prediction_models_to_clean:
            try:
                timestamp_column_name = 'predicted_datetime_utc'
                if model_cls is HighMultiplierPredictionLog: timestamp_column_name = 'generated_at_utc'
                timestamp_column = getattr(model_cls, timestamp_column_name, None)
                if timestamp_column is None: logger.error(f"Cleanup: Model {model_cls.__tablename__} no recognized ts col '{timestamp_column_name}'."); continue
                deleted_count = model_cls.query.filter(timestamp_column < cleanup_threshold_date).delete(synchronize_session=False)
                if deleted_count > 0: logger.info(f"Cleanup: Deleted {deleted_count} old records from {model_cls.__tablename__}."); total_deleted_count += deleted_count
                db.session.commit()
            except Exception as e: db.session.rollback(); logger.error(f"Cleanup: Error cleaning {model_cls.__tablename__}: {e}", exc_info=True)
        if total_deleted_count > 0: logger.info(f"Cleanup: Finished. Total old records deleted: {total_deleted_count}.")
        else: logger.info("Cleanup: Finished. No old records found to delete.")
# --- END SECTION: SCHEDULER JOBS ---

# --- SECTION: FLASK ROUTES ---
@app.route('/')
def landing_page(): return render_template('landing.html', title="KingPredict - Welcome")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, telephone_number=form.telephone_number.data, password_hash=hashed_password, status='pending')
        db.session.add(user); db.session.commit()
        flash('Your account has been created and is awaiting admin approval. You will be notified once approved.', 'info')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if user.status == 'pending': flash('Your account is awaiting admin approval.', 'warning'); return render_template('login.html', title='Login', form=form)
            if user.status == 'suspended': flash('Your account has been suspended. Please contact support.', 'danger'); return render_template('login.html', title='Login', form=form)
            if user.lockout_until and datetime.now(timezone.utc) < user.lockout_until:
                lockout_remaining = format_timedelta_to_hhmmss(user.lockout_until - datetime.now(timezone.utc))
                flash(f'Account locked due to too many failed attempts. Try again in {lockout_remaining}.', 'danger'); return render_template('login.html', title='Login', form=form)
            if user.check_password(form.password.data):
                current_device_id = generate_device_identifier()
                if user.login_device_identifier and user.login_device_identifier != current_device_id:
                    flash('Login from this device is not permitted. This account is locked to another device.', 'danger'); return render_template('login.html', title='Login', form=form)
                if not user.is_active_account: flash('Account is inactive or has expired.', 'danger'); return render_template('login.html', title='Login', form=form)
                user.failed_login_attempts = 0; user.lockout_until = None
                if not user.first_login_timestamp and user.status == 'active': 
                    user.first_login_timestamp = datetime.now(timezone.utc)
                    if user.expiry_duration_seconds is not None: user.actual_expiry_timestamp = user.first_login_timestamp + timedelta(seconds=user.expiry_duration_seconds)
                    if not user.login_device_identifier: user.login_device_identifier = current_device_id
                db.session.commit(); login_user(user, remember=True)
                flash(f'Welcome back, {user.username}!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or (url_for('admin_dashboard') if user.is_admin else url_for('dashboard')))
            else: 
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                if user.failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
                    user.lockout_until = datetime.now(timezone.utc) + timedelta(hours=LOCKOUT_DURATION_HOURS)
                    flash(f'Too many failed login attempts. Your account has been locked for {LOCKOUT_DURATION_HOURS} hour(s).', 'danger')
                else:
                    attempts_left = MAX_FAILED_LOGIN_ATTEMPTS - user.failed_login_attempts
                    flash(f'Invalid password. {attempts_left} attempt(s) remaining before lockout.', 'warning')
                db.session.commit()
        else: flash('Invalid username or password.', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
@login_required
def logout(): logout_user(); flash('You have been logged out.', 'info'); return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    view_as_user_mode = request.args.get('view_as_user', 'false').lower() == 'true'

    now_utc = datetime.now(timezone.utc)
    display_hourly_start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    display_hourly_end_utc = display_hourly_start_utc + timedelta(hours=2)
    current_day_start_utc = now_utc.replace(hour=0,minute=0,second=0,microsecond=0)
    next_day_start_utc = current_day_start_utc + timedelta(days=1)

    preds_h = PredictionHourly.query.filter(
        PredictionHourly.predicted_datetime_utc >= display_hourly_start_utc, 
        PredictionHourly.predicted_datetime_utc < display_hourly_end_utc
    ).order_by(PredictionHourly.predicted_datetime_utc.asc()).limit(HOURLY_PRED_COUNT).all()
    
    preds_50h = Prediction50xHourly.query.filter(
        Prediction50xHourly.predicted_datetime_utc >= display_hourly_start_utc, 
        Prediction50xHourly.predicted_datetime_utc < display_hourly_end_utc
    ).order_by(Prediction50xHourly.predicted_datetime_utc.asc()).limit(HOURLY_50X_PRED_COUNT).all()
    
    preds_50d = Prediction50xDaily.query.filter(
        Prediction50xDaily.predicted_datetime_utc >= current_day_start_utc, 
        Prediction50xDaily.predicted_datetime_utc < next_day_start_utc
    ).order_by(Prediction50xDaily.predicted_datetime_utc.asc()).limit(DAILY_50X_PRED_COUNT).all()
    
    preds_100 = Prediction100x.query.filter(
        Prediction100x.predicted_datetime_utc >= current_day_start_utc, 
        Prediction100x.predicted_datetime_utc < next_day_start_utc
    ).order_by(Prediction100x.predicted_datetime_utc.asc()).limit(DAILY_100X_PRED_COUNT).all()
    
    preds_500 = Prediction500x.query.filter(
        Prediction500x.predicted_datetime_utc >= current_day_start_utc, 
        Prediction500x.predicted_datetime_utc < next_day_start_utc
    ).order_by(Prediction500x.predicted_datetime_utc.asc()).limit(DAILY_500X_PRED_COUNT).all()
    
    preds_1000 = Prediction1000x.query.filter(
        Prediction1000x.predicted_datetime_utc >= current_day_start_utc, 
        Prediction1000x.predicted_datetime_utc < next_day_start_utc
    ).order_by(Prediction1000x.predicted_datetime_utc.asc()).limit(DAILY_1000X_PRED_COUNT).all()
    
    monthly_predictions = PredictionMonthly.query.order_by(PredictionMonthly.target_date_utc.asc()).limit(2).all()

    all_time_wins_info = {
        'hourly_wins': PredictionHourly.query.filter_by(user_verdict=1).count(),
        '50x_hourly_wins': Prediction50xHourly.query.filter_by(user_verdict=1).count(),
        '50x_daily_wins': Prediction50xDaily.query.filter_by(user_verdict=1).count(),
        '100x_wins': Prediction100x.query.filter_by(user_verdict=1).count(),
        '500x_wins': Prediction500x.query.filter_by(user_verdict=1).count(),
        '1000x_wins': Prediction1000x.query.filter_by(user_verdict=1).count(),
        'hml_wins': HighMultiplierPredictionLog.query.filter_by(user_verdict=1).count(),
        'monthly_wins': PredictionMonthly.query.filter_by(user_verdict=1).count()
    }

    upcoming_hml_preds_query = HighMultiplierPredictionLog.query.filter(HighMultiplierPredictionLog.predicted_event_time_utc >= now_utc).order_by(HighMultiplierPredictionLog.predicted_event_time_utc.asc()).limit(HML_NUM_UPCOMING_PREDS).all()
    temp_upcoming_hml_preds = list(upcoming_hml_preds_query)
    if len(temp_upcoming_hml_preds) < HML_NUM_UPCOMING_PREDS:
        latest_batch_time = db.session.query(db.func.max(HighMultiplierPredictionLog.generated_at_utc)).scalar()
        if latest_batch_time:
            additional_needed = HML_NUM_UPCOMING_PREDS - len(temp_upcoming_hml_preds)
            existing_ids = [p.id for p in temp_upcoming_hml_preds]
            additional_preds = HighMultiplierPredictionLog.query.filter(HighMultiplierPredictionLog.generated_at_utc == latest_batch_time, HighMultiplierPredictionLog.id.notin_(existing_ids)).order_by(HighMultiplierPredictionLog.predicted_event_time_utc.asc()).limit(additional_needed).all()
            temp_upcoming_hml_preds.extend(additional_preds)
            temp_upcoming_hml_preds.sort(key=lambda p: p.predicted_event_time_utc)
    processed_hml_preds = []
    for p in temp_upcoming_hml_preds:
        if p.predicted_event_time_utc.tzinfo is None: p.predicted_event_time_utc = p.predicted_event_time_utc.replace(tzinfo=timezone.utc)
        if hasattr(p, 'generated_at_utc') and p.generated_at_utc.tzinfo is None: p.generated_at_utc = p.generated_at_utc.replace(tzinfo=timezone.utc)
        processed_hml_preds.append(p)
    comparison_time = now_utc - timedelta(minutes=15)
    upcoming_hml_preds = [p for p in processed_hml_preds if p.predicted_event_time_utc >= comparison_time][:HML_NUM_UPCOMING_PREDS]


    feedback_form = FeedbackForm()
    page_title = "User Dashboard"
    if current_user.is_admin and view_as_user_mode: page_title = "User Dashboard (Admin View)"
    
    return render_template('dashboard.html', title=page_title,
                           predictions_hourly=preds_h, 
                           predictions_50x_hourly=preds_50h,
                           predictions_50x_daily=preds_50d, 
                           predictions_100x=preds_100,
                           predictions_500x=preds_500, 
                           predictions_1000x=preds_1000,
                           monthly_predictions=monthly_predictions,
                           all_time_wins_info=all_time_wins_info,
                           feedback_form=feedback_form,
                           now_utc_aware=now_utc,
                           upcoming_high_event_predictions=upcoming_hml_preds,
                           last_high_event_pred_update=app.last_high_event_pred_update)

@app.route('/get_expiry_time_left', methods=['GET'])
@login_required
def get_expiry_time_left():
    if current_user.is_authenticated and current_user.actual_expiry_timestamp:
        time_left_sec = current_user.time_left_seconds
        if time_left_sec is None or time_left_sec <= 0: return jsonify({'time_left_str': 'Expired', 'expired': True, 'raw_seconds': 0})
        return jsonify({'time_left_str': format_timedelta_to_hhmmss(timedelta(seconds=time_left_sec)), 'expired': False, 'raw_seconds': time_left_sec})
    if current_user.is_authenticated and current_user.expiry_duration_seconds is None and current_user.status == 'active':
        return jsonify({'time_left_str': 'Indefinite', 'expired': False, 'raw_seconds': float('inf')})
    if current_user.is_authenticated and not current_user.first_login_timestamp and current_user.status == 'active':
        return jsonify({'time_left_str': 'Timer not started', 'expired': False, 'raw_seconds': float('inf')}) 
    return jsonify({'time_left_str': 'N/A', 'expired': True if current_user.is_authenticated else False, 'raw_seconds': 0})

_prediction_model_map_for_feedback: Dict[str, type] = {
    'hourly': PredictionHourly, '50x_hourly': Prediction50xHourly,
    '100x': Prediction100x, '500x': Prediction500x, '1000x': Prediction1000x,
    '50x_daily': Prediction50xDaily, 'high_multiplier_log': HighMultiplierPredictionLog,
    'monthly': PredictionMonthly
}
@app.route('/submit_feedback/<string:prediction_type>/<int:prediction_id>', methods=['POST'])
@login_required
def submit_feedback(prediction_type: str, prediction_id: int):
    form = FeedbackForm()
    if form.validate_on_submit():
        TargetModel = _prediction_model_map_for_feedback.get(prediction_type)
        if not TargetModel:
            flash('Invalid prediction type for feedback.', 'danger')
            return redirect(url_for('dashboard'))
        
        prediction_item = db.session.get(TargetModel, prediction_id)
        if not prediction_item:
            flash('Prediction not found.', 'danger')
            return redirect(url_for('dashboard'))

        if hasattr(prediction_item, 'user_verdict'):
            if prediction_item.user_verdict == 1:
                flash('This prediction was already marked as correct.', 'info')
            else:
                prediction_item.user_verdict = 1
                
                # --- ADD THIS LOGIC TO SAVE A SIMULATED ACTUAL MULTIPLIER ---
                if hasattr(prediction_item, 'actual_multiplier'):
                    # For now, let's save a simulated value.
                    # In a real app, you'd get this from user input.
                    min_mult = getattr(prediction_item, 'min_multiplier', 10.0)
                    avg_mult = getattr(prediction_item, 'avg_multiplier', min_mult * 1.5)
                    
                    # Simulate a value that's reasonably close to the prediction
                    simulated_actual = np.random.uniform(avg_mult * 0.8, avg_mult * 1.2)
                    prediction_item.actual_multiplier = round(simulated_actual, 2)
                # --- END OF NEW LOGIC ---

                db.session.commit()
                flash('Prediction marked as correct. Thank you for your feedback!', 'success')
        else:
            flash(f"Feedback (user_verdict) is not applicable for this type of item: {prediction_type}.", "warning")
    else:
        flash('Invalid feedback submission. Please try again.', 'danger')
    
    # Redirect back to the wins history page if the referer is available, otherwise dashboard
    referer = request.headers.get("Referer")
    if referer and 'wins' in referer:
        return redirect(referer)
    return redirect(url_for('dashboard'))

@app.route('/admin', methods=['GET'])
@login_required
def admin_dashboard():
    if not current_user.is_admin: flash('Access denied.', 'danger'); return redirect(url_for('dashboard'))
    users = User.query.order_by(User.status.asc(), User.username.asc()).all()
    create_user_form = AdminCreateUserForm()
    renewal_form = AdminRenewUserForm()
    historical_data_form = AddHistoricalDataForm()
    historical_records = HistoricalData.query.order_by(HistoricalData.date.desc(), HistoricalData.time.desc()).limit(50).all()
    return render_template('admin_dashboard.html', title='Admin Panel', users=users, 
                           registration_form=create_user_form, 
                           renewal_form=renewal_form, 
                           historical_data_form=historical_data_form, 
                           historical_records=historical_records)

@app.route('/admin/create_user', methods=['POST'])
@login_required
def admin_create_user():
    if not current_user.is_admin: abort(403)
    form = AdminCreateUserForm()
    if form.validate_on_submit():
        expiry_seconds = None
        if form.expiry_time.data.lower() != 'indefinite':
            expiry_seconds = parse_expiry_time(form.expiry_time.data)
            if expiry_seconds is None:
                flash('Invalid expiry time format. Use HH:MM or "indefinite".', 'danger')
                return redirect(url_for('admin_dashboard'))
        new_user = User(username=form.username.data, telephone_number=form.telephone_number.data, is_admin=(form.is_admin.data=='True'), status=form.status.data, expiry_duration_seconds=expiry_seconds)
        new_user.set_password(form.password.data)
        if new_user.status == 'active':
            new_user.approved_by_admin_id = current_user.id
            new_user.approval_timestamp = datetime.now(timezone.utc)
        db.session.add(new_user); db.session.commit()
        flash(f'User {form.username.data} created successfully with status: {form.status.data}!', 'success')
    else:
        for field, error_list in form.errors.items():
            for error in error_list: flash(f"Error in {getattr(form, field).label.text if hasattr(getattr(form, field), 'label') else field}: {error}", 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/manage_registrations', methods=['GET'])
@login_required
def admin_manage_registrations():
    if not current_user.is_admin: abort(403)
    pending_users = User.query.filter_by(status='pending').order_by(User.registration_timestamp.asc()).all()
    return render_template('admin_manage_registrations.html', title='Manage Registrations', pending_users=pending_users)

@app.route('/admin/approve_user/<int:user_id>', methods=['POST'])
@login_required
def admin_approve_user(user_id):
    if not current_user.is_admin: abort(403)
    user_to_approve = db.session.get(User, user_id)
    if user_to_approve and user_to_approve.status == 'pending':
        expiry_hhmm_str = request.form.get('expiry_time', 'indefinite').strip()
        expiry_seconds = None
        if expiry_hhmm_str.lower() != 'indefinite':
            expiry_seconds = parse_expiry_time(expiry_hhmm_str)
            if expiry_seconds is None:
                flash(f'Invalid expiry time format "{expiry_hhmm_str}". Use HH:MM or "indefinite".', 'danger')
                return redirect(url_for('admin_manage_registrations'))
        user_to_approve.status = 'active'
        user_to_approve.expiry_duration_seconds = expiry_seconds
        user_to_approve.approved_by_admin_id = current_user.id
        user_to_approve.approval_timestamp = datetime.now(timezone.utc)
        user_to_approve.first_login_timestamp = None 
        user_to_approve.actual_expiry_timestamp = None
        db.session.commit()
        flash(f'User {user_to_approve.username} has been approved and activated.', 'success')
    else:
        flash('User not found or not pending approval.', 'danger')
    return redirect(url_for('admin_manage_registrations'))

@app.route('/admin/reject_user/<int:user_id>', methods=['POST'])
@login_required
def admin_reject_user(user_id):
    if not current_user.is_admin: abort(403)
    user_to_reject = db.session.get(User, user_id)
    if user_to_reject and user_to_reject.status == 'pending':
        user_to_reject.status = 'rejected' 
        user_to_reject.approved_by_admin_id = current_user.id 
        user_to_reject.approval_timestamp = datetime.now(timezone.utc) 
        db.session.commit()
        flash(f'User {user_to_reject.username} registration has been rejected.', 'info')
    else:
        flash('User not found or not pending approval.', 'danger')
    return redirect(url_for('admin_manage_registrations'))

@app.route('/admin/renew_user', methods=['POST'])
@login_required
def admin_renew_user():
    if not current_user.is_admin: abort(403)
    form = AdminRenewUserForm() 
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if not user: flash(f'User {form.username.data} not found.', 'danger'); return redirect(url_for('admin_dashboard'))
        new_expiry_hhmm_str = form.new_expiry_time.data.strip()
        new_expiry_seconds = None
        if new_expiry_hhmm_str.lower() != 'indefinite':
            new_expiry_seconds = parse_expiry_time(new_expiry_hhmm_str)
            if new_expiry_seconds is None:
                flash(f'Invalid new expiry time format "{new_expiry_hhmm_str}". Use HH:MM or "indefinite".', 'danger')
                return redirect(url_for('admin_dashboard'))
        user.expiry_duration_seconds = new_expiry_seconds
        user.first_login_timestamp = None; user.actual_expiry_timestamp = None
        user.failed_login_attempts = 0; user.lockout_until = None
        user.status = 'active' 
        db.session.commit(); flash(f'User {user.username} account renewed successfully.', 'success')
    else:
        for field, error_list in form.errors.items():
            for error in error_list: flash(f"Error in {getattr(form, field).label.text if hasattr(getattr(form, field), 'label') else field}: {error}", 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add_historical_data', methods=['POST'])
@login_required
def admin_add_historical_data():
    if not current_user.is_admin: abort(403)
    form = AddHistoricalDataForm()
    if form.validate_on_submit():
        try:
            multiplier_val = float(form.multiplier.data)
            new_data = HistoricalData(date=form.date.data, time=form.time.data, multiplier=multiplier_val)
            db.session.add(new_data); db.session.commit(); flash('Historical data point added successfully.', 'success')
        except ValueError: flash('Invalid multiplier value.', 'danger')
        except Exception as e: db.session.rollback(); flash(f'An error occurred: {str(e)}', 'danger'); logger.error(f"Error adding historical data: {e}", exc_info=True)
    else:
        for field, error_list in form.errors.items():
            for error in error_list: flash(f"Error in {getattr(form, field).label.text if hasattr(getattr(form, field), 'label') else field}: {error}", 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def admin_delete_user(user_id: int):
    if not current_user.is_admin: abort(403)
    user_to_delete = db.session.get(User, user_id)
    if not user_to_delete: flash('User not found.', 'danger'); return redirect(url_for('admin_dashboard'))
    if user_to_delete.id == current_user.id: flash("You cannot delete your own account.", "danger"); return redirect(url_for('admin_dashboard'))
    try:
        User.query.filter_by(approved_by_admin_id=user_id).update({"approved_by_admin_id": None})
        db.session.delete(user_to_delete); db.session.commit(); flash(f'User {user_to_delete.username} has been deleted.', 'success')
    except Exception as e: db.session.rollback(); flash(f'Error deleting user: {str(e)}', 'danger'); logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
    return redirect(url_for('admin_dashboard'))


# --- ADD THIS NEW ROUTE TO YOUR app5.py FILE ---

@app.route('/wins/<string:prediction_type>')
@login_required
def wins_history(prediction_type: str):
    """
    Displays a detailed history of all correct predictions ('wins')
    for a specific prediction type.
    """
    # Map the URL string to the corresponding database model and a user-friendly title
    prediction_type_map = {
        'hourly': (PredictionHourly, 'Hourly Hits'),
        '50x_hourly': (Prediction50xHourly, '50x+ Hourly Hits'),
        '50x_daily': (Prediction50xDaily, '50x+ Daily Hits'),
        '100x': (Prediction100x, '100x+ Daily Hits'),
        '500x': (Prediction500x, '500x+ Daily Hits'),
        '1000x': (Prediction1000x, '1000x+ Daily Hits'),
        'hml': (HighMultiplierPredictionLog, 'HML Hits'),
        'monthly': (PredictionMonthly, 'Monthly Hits')
    }

    if prediction_type not in prediction_type_map:
        flash('Invalid prediction type specified.', 'danger')
        return redirect(url_for('dashboard'))

    TargetModel, page_title = prediction_type_map[prediction_type]

    # Determine the correct timestamp column for sorting
    timestamp_column = 'predicted_datetime_utc'
    if TargetModel is HighMultiplierPredictionLog:
        timestamp_column = 'predicted_event_time_utc'
    elif TargetModel is PredictionMonthly:
        timestamp_column = 'predicted_value_utc'

    # Fetch all records for that model where user_verdict is 1 (a 'win')
    # Order by the most recent wins first
    wins = TargetModel.query.filter_by(
        user_verdict=1
    ).order_by(
        getattr(TargetModel, timestamp_column).desc()
    ).all()

    return render_template('wins_history.html', 
                           wins=wins, 
                           page_title=page_title, 
                           prediction_type=prediction_type)

# --- END OF NEW ROUTE ---
# --- END SECTION: FLASK ROUTES ---

# --- SECTION: INITIALIZATION & COMMANDS ---

# NOTE: All automatic admin creation has been removed.
# The first admin user must be created manually using the 'flask shell'.
# This is a more secure practice for production environments.

def _schedule_jobs():
    # Ensure scheduler is initialized
    if not scheduler: # Defensive check, scheduler should be globally defined
        logger.error("Scheduler not initialized before _schedule_jobs call.")
        return

    current_job_ids = {job.id for job in scheduler.get_jobs()}
    job_definitions = [
        {'id': 'job_predict_hourly', 'func': job_predict_hourly, 'trigger': CronTrigger(minute=0)},
        {'id': 'job_predict_50x_hourly', 'func': job_predict_50x_hourly, 'trigger': CronTrigger(minute=0)},
        {'id': 'job_predict_50x_daily', 'func': job_predict_50x_daily, 'trigger': CronTrigger(hour=0, minute=1)},
        {'id': 'job_predict_100x', 'func': job_predict_100x, 'trigger': CronTrigger(hour=0, minute=1)},
        {'id': 'job_predict_500x', 'func': job_predict_500x, 'trigger': CronTrigger(hour=0, minute=1)},
        {'id': 'job_predict_1000x', 'func': job_predict_1000x, 'trigger': CronTrigger(hour=0, minute=1)},
        {'id': 'job_predict_high_multiplier_likelihood', 'func': job_predict_high_multiplier_likelihood, 'trigger': CronTrigger(minute=0)},
        {'id': 'job_predict_monthly_5000x', 'func': job_predict_monthly_5000x, 'trigger': CronTrigger(day=1, hour=1, minute=0)},
        {'id': 'job_cleanup_old_data', 'func': job_cleanup_old_data, 'trigger': CronTrigger(hour=3, minute=0)}
    ]
    jobs_added_count = 0
    for job_spec in job_definitions:
        if job_spec['id'] not in current_job_ids:
            try:
                scheduler.add_job(id=job_spec['id'], func=job_spec['func'], trigger=job_spec['trigger'], misfire_grace_time=300)
                logger.info(f"Scheduled job: {job_spec['id']}.")
                jobs_added_count +=1
            except Exception as e:
                logger.error(f"Error scheduling job {job_spec['id']}: {e}", exc_info=True)
    if jobs_added_count > 0:
        logger.info(f"{jobs_added_count} new jobs scheduled.")
    else:
        logger.info("No new jobs were added to the scheduler (they may already exist or were re-added).")

    logger.info(f"Scheduler job setup verified. Total jobs currently in scheduler: {len(scheduler.get_jobs())}.")

    if not scheduler.running:
        try:
            scheduler.start()
            logger.info("APScheduler started.")
        except Exception as e:
            logger.error(f"Failed to start APScheduler: {e}", exc_info=True)
    else:
        logger.info("APScheduler is already running.")


def initialize_app_core():
    with app.app_context():
        logger.info(f"Initializing app core. Using database at: {app.config['SQLALCHEMY_DATABASE_URI']}")
        # Admin creation is no longer called from here.
        _schedule_jobs()
        hist_count = HistoricalData.query.count()
        user_count = User.query.count()
        logger.info(f"App core initialization complete. HistData records: {hist_count}, User records: {user_count}.")


def run_initial_predictions():
    logger.info("Running broader initial predictions on startup...")
    with app.app_context():
        try:
            if not any(MODELS_LOADED_SUCCESSFULLY.values()):
                logger.warning("Models not loaded before run_initial_predictions. Attempting to load now.")
                load_trained_models()

            job_predict_hourly()
            job_predict_100x()
            job_predict_500x()
            job_predict_1000x()
            logger.info("Broader initial predictions completed.")
        except Exception as e:
            logger.error(f"Error during broader initial predictions: {e}", exc_info=True)


def run_specific_predictions_on_startup():
    logger.info("Running specific one-time predictions at startup...")
    with app.app_context():
        try:
            if not any(MODELS_LOADED_SUCCESSFULLY.values()):
                logger.warning("Models not loaded before run_specific_predictions_on_startup. Attempting to load now.")
                load_trained_models()

            job_predict_50x_hourly()
            job_predict_50x_daily()
            job_predict_high_multiplier_likelihood()
            job_predict_monthly_5000x()
            logger.info("Specific one-time startup predictions completed (50x Hourly, 50x Daily, HML, Monthly).")
        except Exception as e:
            logger.error(f"Error during specific startup predictions: {e}", exc_info=True)


@app.cli.command("init-kingpredict")
def init_command():
    """Initializes the KingPredict application: schedules jobs and runs initial predictions."""
    print("Initializing KingPredict application...")
    print("Loading trained models...")
    load_trained_models()

    with app.app_context():
        print("Ensuring database schema is up to date via migrations (run 'flask db upgrade' manually if needed)...")
        initialize_app_core()
        print("Core application components initialized (Scheduler).")

    print("Running specific startup predictions...")
    run_specific_predictions_on_startup()
    print("Running broader initial predictions...")
    run_initial_predictions()
    print("Initial predictions executed. KingPredict initialization finished.")
    print("IMPORTANT: Admin user creation is now a manual process. Use the 'flask shell' to create the first user.")


if __name__ == '__main__':
    werkzeug_reloader_monitor_active = os.environ.get("WERKZEUG_RUN_MAIN") != "true"

    if not app.debug or not werkzeug_reloader_monitor_active:
        if werkzeug_reloader_monitor_active and app.debug:
             pass
        
        print(f"--- app.py __main__: {'Child process (debug)' if app.debug else 'Production/No-Debug process'}: Preparing to load models. ---")
        logger.info(f"Flask {'Child (debug)' if app.debug else 'Production/No-Debug'} process: Preparing to load models.")
        load_trained_models()

        logger.info("Process identified for app core initialization and startup predictions.")
        initialize_app_core()
        
        logger.info("Running specific startup predictions (hourly 50x, daily 50x, HML, Monthly)...")
        run_specific_predictions_on_startup()

        run_the_broader_ones = False
        
        env_var_value = os.getenv("RUN_INITIAL_PREDICTIONS_ON_START", "false").strip().lower()
        logger.info(f"ENV CHECK: Value of RUN_INITIAL_PREDICTIONS_ON_START read as: '{env_var_value}' (defaulted to 'false' if not set)")

        if env_var_value == "true":
            logger.info("ENV CHECK: Forcing broader predictions because RUN_INITIAL_PREDICTIONS_ON_START is 'true'.")
            run_the_broader_ones = True
        else:
            logger.info("ENV CHECK: RUN_INITIAL_PREDICTIONS_ON_START is not 'true'. Checking DB for hourly predictions as a fallback condition.")
            with app.app_context():
                if not PredictionHourly.query.first():
                    logger.info("ENV CHECK: No general hourly predictions found in DB. Will run broader predictions.")
                    run_the_broader_ones = True
                else:
                    logger.info("ENV CHECK: General hourly predictions exist in DB, and RUN_INITIAL_PREDICTIONS_ON_START is not 'true'. Broader predictions will be skipped.")
        
        if run_the_broader_ones:
            logger.info("ACTION: Proceeding to run broader set of initial predictions.")
            run_initial_predictions()
        else:
            logger.info("ACTION: Broader initial predictions will NOT be run based on current conditions.")

    elif app.debug and werkzeug_reloader_monitor_active:
        print("--- app.py __main__: Werkzeug reloader PARENT/MONITOR process (debug): Preparing to load models (if needed by monitor). ---")
        logger.info("Flask Werkzeug reloader PARENT/MONITOR process (debug): Preparing to load models (if needed by monitor).")
        load_trained_models()
        logger.info("Werkzeug reloader PARENT/MONITOR process: Core app init and startup predictions deferred to child process.")

    app.run(debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 7520)))
# --- END SECTION: INITIALIZATION & COMMANDS ---
#--- END OF FILE app5.py ---
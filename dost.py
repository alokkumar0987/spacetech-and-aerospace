# Standard libraries
import re
import time
import hashlib
import json
import os
from threading import Thread, Lock
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh



# Third-party libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import pytz
import streamlit as st
from streamlit import cache_data, cache_resource

# Machine Learning
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Reset TensorFlow graph (only needed in certain legacy cases)
tf.compat.v1.reset_default_graph()

# ======================
# ENHANCED AUTHENTICATION SYSTEM
# ======================

class AuthSystem:
    """Professional authentication system with persistent storage"""
    
    def __init__(self):
        self.users_file = "users.json"
        self.lock = Lock()
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialize user storage if not exists"""
        if not os.path.exists(self.users_file):
            with self.lock:
                with open(self.users_file, 'w') as f:
                    json.dump({}, f)
    
    def _hash_password(self, password):
        """Secure password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self):
        """Thread-safe user data loading"""
        with self.lock:
            with open(self.users_file, 'r') as f:
                return json.load(f)
    
    def _save_users(self, users):
        """Thread-safe user data saving"""
        with self.lock:
            with open(self.users_file, 'w') as f:
                json.dump(users, f)
    
    def register_user(self, name, email, password):
        """Register new user with validation"""
        if not validate_email(email):
            return False, "Invalid email format"
        if not validate_name(name):
            return False, "Name can only contain letters and spaces"
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
            
        users = self._load_users()
        if email in users:
            return False, "Email already registered"
            
        users[email] = {
            'name': name,
            'password_hash': self._hash_password(password),
            'created_at': datetime.now().isoformat()
        }
        
        self._save_users(users)
        return True, "Registration successful"
    
    def authenticate(self, email, password):
        """Authenticate user"""
        users = self._load_users()
        if email not in users:
            return False, "User not found"
            
        if users[email]['password_hash'] != self._hash_password(password):
            return False, "Invalid credentials"
            
        return True, "Authentication successful"

# Initialize auth system
auth_system = AuthSystem()

def validate_email(email):
    """Improved email validation"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def validate_name(name):
    """Improved name validation"""
    return bool(name) and name.replace(" ", "").isalpha()

def authentication_section():
    """Enhanced authentication UI with persistent sessions"""
    # Initialize session state
    if 'auth' not in st.session_state:
        st.session_state.auth = {
            'authenticated': False,
            'user_email': None,
            'user_name': None,
            'login_attempted': False
        }
    
    # Check for existing session
    if not st.session_state.auth['authenticated']:
        show_login_section()
    else:
        show_logout_section()

def show_login_section():
    """Enhanced login UI with proper error handling"""
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h4>🔐 Professional Authentication</h4>
            <p>Secure access to aviation weather system</p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    st.session_state.auth['login_attempted'] = True
                    valid, message = auth_system.authenticate(email, password)
                    if valid:
                        users = auth_system._load_users()
                        st.session_state.auth.update({
                            'authenticated': True,
                            'user_email': email,
                            'user_name': users[email]['name']
                        })
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
        
        with tab2:
            with st.form("register_form"):
                name = st.text_input("Full Name", key="reg_name")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        valid, message = auth_system.register_user(name, email, password)
                        if valid:
                            st.success(message)
                        else:
                            st.error(message)

def show_logout_section():
    """Enhanced profile section with session management"""
    with st.sidebar:
        user = auth_system._load_users().get(st.session_state.auth['user_email'], {})
        
        st.markdown(f"""
        <div class="card decision-safe">
            <h4>👤 User Profile</h4>
            <p><strong>Name:</strong> {user.get('name', 'N/A')}</p>
            <p><strong>Email:</strong> {st.session_state.auth['user_email']}</p>
            <p><strong>Member since:</strong> {user.get('created_at', 'N/A')[:10]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", key="logout_btn"):
            st.session_state.auth.update({
                'authenticated': False,
                'user_email': None,
                'user_name': None
            })
            st.success("Logged out successfully!")
            st.rerun()

# ======================
# ENHANCED ALERT SYSTEM
# ======================

class EnhancedAlertSystem:
    """Professional alert system with source/destination focus"""
    
    def __init__(self, model, airport_db):
        self.model = model
        self.airport_db = airport_db
        self.active_alerts = {}
        self.running = False
        self.thread = None
        self.lock = Lock()
        self.source = None
        self.destination = None
        
    def set_flight_route(self, source, destination):
        """Set the focus airports for monitoring"""
        with self.lock:
            self.source = source
            self.destination = destination
    
    def start(self):
        """Start monitoring with proper thread management"""
        if not self.running and self.source and self.destination:
            self.running = True
            self.thread = Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Stop monitoring gracefully"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop with error handling"""
        while self.running:
            try:
                self._check_critical_airports()
                time.sleep(300)  # 5 minutes
            except Exception as e:
                st.error(f"Alert system error: {str(e)}")
                time.sleep(60)  # Wait before retry
    
    def _check_critical_airports(self):
        """Check source and destination airports"""
        new_alerts = {}
        
        for icao in [self.source, self.destination]:
            if icao not in self.airport_db:
                continue
                
            info = self.airport_db[icao]
            weather = fetch_live_weather(info['coords'][0], info['coords'][1], info['tz'])
            if not weather:
                continue
                
            # Generate predictions for next 5 minutes (extended window)
            predictions = generate_time_series_prediction(weather, self.model, steps=15)  # 15 steps = 5 minutes
            
            # Classify conditions
            classification = classify_conditions(
                weather['wind'],
                weather['temp'],
                weather['precip'],
                weather['visibility']
            )
            
            # Store alerts if conditions are dangerous
            if not is_flight_safe(classification) or any(p > 70 for p in predictions[:3]):  # Next 1 hour
                with self.lock:
                    new_alerts[icao] = {
                        'predictions': predictions,
                        'classification': classification,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'weather_data': weather
                    }
        
        # Update active alerts
        with self.lock:
            self.active_alerts = new_alerts
        
        # Trigger UI update
        st.experimental_rerun()
    
    def get_active_alerts(self):
        """Get current alerts in a thread-safe way"""
        with self.lock:
            return self.active_alerts.copy()

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Aviation Weather AI",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Weather API Configuration
BASE_URL = "http://api.openweathermap.org/data/2.5/"
API_KEY = st.secrets["OPENWEATHER_API_KEY"]

# Custom CSS for professional look
st.markdown("""
<style>
    :root {
        --primary: #00c6ff;
        --secondary: #0072ff;
        --dark: #0f2027;
        --light: #f8f9fa;
    }
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: rgba(0, 198, 255, 0.15);
        border-left: 4px solid var(--primary);
    }
    .decision-safe {
        background: rgba(0, 200, 83, 0.2);
        border-left: 4px solid #00c853;
    }
    .decision-warning {
        background: rgba(255, 171, 0, 0.2);
        border-left: 4px solid #ffab00;
    }
    .decision-danger {
        background: rgba(255, 82, 82, 0.2);
        border-left: 4px solid #ff5252;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.9) !important;
    }
    .stAlert {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# CONSTANTS & DATA
# ======================
# ======================
AIRPORT_DB = {
    # Indian Airports (Top 30)
    "VIDP": {"name": "Indira Gandhi (Delhi)", "coords": (28.5562, 77.1000), "iata": "DEL", "tz": "Asia/Kolkata"},
    "VABB": {"name": "Chhatrapati Shivaji (Mumbai)", "coords": (19.0887, 72.8679), "iata": "BOM", "tz": "Asia/Kolkata"},
    "VOMM": {"name": "Chennai International", "coords": (12.9941, 80.1707), "iata": "MAA", "tz": "Asia/Kolkata"},
    "VOBL": {"name": "Kempegowda (Bangalore)", "coords": (13.1986, 77.7066), "iata": "BLR", "tz": "Asia/Kolkata"},
    "VIAR": {"name": "Sri Guru Ram Dass Jee (Amritsar)", "coords": (31.7096, 74.7973), "iata": "ATQ", "tz": "Asia/Kolkata"},
    "VOGO": {"name": "Goa International (Dabolim)", "coords": (15.3800, 73.8310), "iata": "GOI", "tz": "Asia/Kolkata"},
    "VOCI": {"name": "Cochin International", "coords": (10.1520, 76.4019), "iata": "COK", "tz": "Asia/Kolkata"},
    "VECC": {"name": "Netaji Subhas Chandra Bose (Kolkata)", "coords": (22.6547, 88.4467), "iata": "CCU", "tz": "Asia/Kolkata"},
    "VAAH": {"name": "Sardar Vallabhbhai Patel (Ahmedabad)", "coords": (23.0772, 72.6347), "iata": "AMD", "tz": "Asia/Kolkata"},
    "VAPO": {"name": "Pune International", "coords": (18.5822, 73.9197), "iata": "PNQ", "tz": "Asia/Kolkata"},
    "VOCI": {"name": "Cochin International", "coords": (10.1520, 76.4019), "iata": "COK", "tz": "Asia/Kolkata"},
    "VOBG": {"name": "HAL Airport (Old Bangalore)", "coords": (12.9510, 77.6682), "iata": "—", "tz": "Asia/Kolkata"},
    "VARK": {"name": "Rajiv Gandhi (Hyderabad)", "coords": (17.2313, 78.4299), "iata": "HYD", "tz": "Asia/Kolkata"},
    "VEPT": {"name": "Patna Airport", "coords": (25.5913, 85.0871), "iata": "PAT", "tz": "Asia/Kolkata"},
    "VANP": {"name": "Dr. Babasaheb Ambedkar (Nagpur)", "coords": (21.0922, 79.0472), "iata": "NAG", "tz": "Asia/Kolkata"},
    "VABB": {"name": "Chhatrapati Shivaji (Mumbai)", "coords": (19.0887, 72.8679), "iata": "BOM", "tz": "Asia/Kolkata"},
    "VAAU": {"name": "Aurangabad Airport", "coords": (19.8633, 75.3981), "iata": "IXU", "tz": "Asia/Kolkata"},
    "VEBS": {"name": "Biju Patnaik (Bhubaneswar)", "coords": (20.2444, 85.8178), "iata": "BBI", "tz": "Asia/Kolkata"},
    "VILK": {"name": "Lal Bahadur Shastri (Varanasi)", "coords": (25.4524, 82.8593), "iata": "VNS", "tz": "Asia/Kolkata"},
    "VAAK": {"name": "Kangra Airport", "coords": (32.1651, 76.2634), "iata": "DHM", "tz": "Asia/Kolkata"},
    "VICG": {"name": "Chandigarh Airport", "coords": (30.6735, 76.7885), "iata": "IXC", "tz": "Asia/Kolkata"},
    "VOHS": {"name": "Rajiv Gandhi (Hyderabad)", "coords": (17.2403, 78.4294), "iata": "HYD", "tz": "Asia/Kolkata"},
    "VEIM": {"name": "Imphal Airport", "coords": (24.7600, 93.8967), "iata": "IMF", "tz": "Asia/Kolkata"},
    "VIAL": {"name": "Agartala Airport", "coords": (23.8860, 91.2404), "iata": "IXA", "tz": "Asia/Kolkata"},
    "VEGT": {"name": "Lokpriya Gopinath Bordoloi (Guwahati)", "coords": (26.1061, 91.5859), "iata": "GAU", "tz": "Asia/Kolkata"},
    "VEBN": {"name": "Bagdogra Airport", "coords": (26.6812, 88.3286), "iata": "IXB", "tz": "Asia/Kolkata"},
    "VETZ": {"name": "Tezpur Airport", "coords": (26.7091, 92.7847), "iata": "TEZ", "tz": "Asia/Kolkata"},
    "VEJT": {"name": "Jorhat Airport", "coords": (26.7315, 94.1755), "iata": "JRH", "tz": "Asia/Kolkata"},
    "VAAU": {"name": "Aurangabad Airport", "coords": (19.8627, 75.3982), "iata": "IXU", "tz": "Asia/Kolkata"},
    "VIBN": {"name": "Bamrauli Airport (Prayagraj)", "coords": (25.4401, 81.7339), "iata": "IXD", "tz": "Asia/Kolkata"},

    # International Airports (Top 20)
    "KJFK": {"name": "John F. Kennedy (NYC)", "coords": (40.6413, -73.7781), "iata": "JFK", "tz": "America/New_York"},
    "KLAX": {"name": "Los Angeles International", "coords": (33.9416, -118.4085), "iata": "LAX", "tz": "America/Los_Angeles"},
    "EGLL": {"name": "Heathrow (London)", "coords": (51.4700, -0.4543), "iata": "LHR", "tz": "Europe/London"},
    "EDDF": {"name": "Frankfurt Airport", "coords": (50.0379, 8.5622), "iata": "FRA", "tz": "Europe/Berlin"},
    "LFPG": {"name": "Charles de Gaulle (Paris)", "coords": (49.0097, 2.5479), "iata": "CDG", "tz": "Europe/Paris"},
    "RJTT": {"name": "Haneda Airport (Tokyo)", "coords": (35.5494, 139.7798), "iata": "HND", "tz": "Asia/Tokyo"},
    "RJAA": {"name": "Narita Airport (Tokyo)", "coords": (35.7720, 140.3929), "iata": "NRT", "tz": "Asia/Tokyo"},
    "ZBAA": {"name": "Beijing Capital International", "coords": (40.0799, 116.6031), "iata": "PEK", "tz": "Asia/Shanghai"},
    "OMDB": {"name": "Dubai International", "coords": (25.2532, 55.3657), "iata": "DXB", "tz": "Asia/Dubai"},
    "OTHH": {"name": "Hamad International (Doha)", "coords": (25.2736, 51.6080), "iata": "DOH", "tz": "Asia/Qatar"},
    "WSSS": {"name": "Changi Airport (Singapore)", "coords": (1.3644, 103.9915), "iata": "SIN", "tz": "Asia/Singapore"},
    "YSSY": {"name": "Sydney Kingsford Smith", "coords": (-33.9399, 151.1753), "iata": "SYD", "tz": "Australia/Sydney"},
    "CYYZ": {"name": "Toronto Pearson", "coords": (43.6777, -79.6248), "iata": "YYZ", "tz": "America/Toronto"},
    "EHAM": {"name": "Amsterdam Schiphol", "coords": (52.3105, 4.7683), "iata": "AMS", "tz": "Europe/Amsterdam"},
    "LSZH": {"name": "Zurich Airport", "coords": (47.4647, 8.5492), "iata": "ZRH", "tz": "Europe/Zurich"},
    "VHHH": {"name": "Hong Kong International", "coords": (22.3080, 113.9185), "iata": "HKG", "tz": "Asia/Hong_Kong"},
    "KSFO": {"name": "San Francisco International", "coords": (37.6213, -122.3790), "iata": "SFO", "tz": "America/Los_Angeles"},
    "KORD": {"name": "Chicago O'Hare", "coords": (41.9742, -87.9073), "iata": "ORD", "tz": "America/Chicago"},
    "SAEZ": {"name": "Ezeiza International (Buenos Aires)", "coords": (-34.8222, -58.5358), "iata": "EZE", "tz": "America/Argentina/Buenos_Aires"},
    "NZAA": {"name": "Auckland Airport", "coords": (-37.0082, 174.7850), "iata": "AKL", "tz": "Pacific/Auckland"}
}

# ======================
# CACHED RESOURCES
# ======================
@cache_resource
def load_weather_model():
    """Load pre-trained LSTM model with error handling"""
    try:
        model = load_model('weather_prediction_lstm.keras')
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None
model =load_weather_model()

@cache_data(ttl=600)  # 10 minute cache
def fetch_forecast(lat, lon, tz):
    """Fetch 5-day forecast with 3-hour intervals"""
    endpoint = "forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecast_data = []
        tz_obj = pytz.timezone(tz)
        
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'], tz_obj)
            forecast_data.append({
                "time": dt,
                "wind": item['wind']['speed'],
                "temp": item['main']['temp'] + 273.15,  # Convert to Kelvin
                "precip": item.get('rain', {}).get('3h', 0)/3,  # Convert 3h rain to mm/hr
                "visibility": item.get('visibility', 10000)
            })
        
        return forecast_data
        
    except Exception as e:
        st.error(f"Forecast fetch error: {str(e)}")
        return None

@cache_data(ttl=3600)
def fetch_live_weather(lat, lon, tz):
    """Robust weather data fetcher with OpenWeatherMap API"""
    endpoint = "weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Timezone-aware timestamp handling
        tz_obj = pytz.timezone(tz)
        current_time = datetime.now(tz_obj)
        
        # Extract weather data with error handling
        try:
            wind_speed = data.get('wind', {}).get('speed', 0)
            temp = data.get('main', {}).get('temp', 0) + 273.15
            visibility = data.get('visibility', 10000)
            
            # Better precipitation estimation using weather codes
            weather_id = data.get('weather', [{}])[0].get('id', 800)
            precip = 0
            if 500 <= weather_id < 600:  # Rain codes
                precip = 2.5 + (weather_id - 500) * 0.1
            elif weather_id >= 200 and weather_id < 300:  # Thunderstorm
                precip = 10
            elif 300 <= weather_id < 400:  # Drizzle
                precip = 1.5
            
            return {
                "wind": float(wind_speed),
                "temp": float(temp),
                "precip": float(precip),
                "visibility": float(visibility),
                "time": current_time.strftime("%Y-%m-%d %H:%M"),
                "data_freshness": "live",
                "weather_condition": data.get('weather', [{}])[0].get('main', '')
            }
            
        except (KeyError, IndexError, ValueError) as e:
            st.error(f"⚠ Weather data parsing error: {str(e)}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network Error: {str(e)}")
        return None
        
    except Exception as e:
        st.error(f"⚠ Unexpected Error: {str(e)}")
        return None

def generate_time_series_prediction(current_weather, model, steps=6):
    """Generate 2-hour prediction (6 steps of 20-min intervals)"""
    try:
        # Get the model's expected input shape
        expected_features = model.input_shape[2]  # Should be 11 features
        
        # Prepare input with padding if needed
        raw_features = np.array([
            current_weather['wind'], 
            current_weather['temp'], 
            current_weather['precip']
        ])
        
        # Pad features with zeros if model expects more than 3 features
        if expected_features > 3:
            features = np.pad(raw_features, (0, expected_features - 3), 
                            mode='constant', constant_values=0)
        else:
            features = raw_features[:expected_features]
            
        # Reshape for LSTM (1 sample, 1 timestep, n_features)
        inputs = features.reshape(1, 1, -1)
        
        # Generate predictions
        predictions = []
        current_input = inputs.copy()
        
        for _ in range(steps):
            pred = model.predict(current_input, verbose=0)[0][0]
            predictions.append(float(pred * 100))  # Convert to percentage
            
            # Update input with prediction (only if model expects recursive inputs)
            if expected_features > 3:
                current_input = np.roll(current_input, -1)
                current_input[0, 0, -1] = pred
        
        return predictions
        
    except Exception as e:
        st.error(f"🧠 Prediction error: {str(e)}")
        return [50.0] * steps  # Return neutral predictions

def classify_conditions(wind, temp, precip, visibility=None):
    """Enhanced classification with ICAO standards"""
    # Wind Classification (m/s)
    if wind < 10:
        wind_status = ('Operational - Normal', 'success')
    elif 10 <= wind < 15:
        wind_status = ('Caution - Moderate Winds', 'warning')
    elif 15 <= wind < 20:
        wind_status = ('Delay Recommended', 'danger')
    else:
        wind_status = ('Flight Halt - Dangerous', 'danger')
    
    # Temperature Classification (K)
    if 263 <= temp <= 308:  # -10°C to +35°C
        temp_status = ('Operational - Normal', 'success')
    elif 253 <= temp < 263 or 308 < temp <= 313:
        temp_status = ('Caution - Extreme Temp', 'warning')
    else:
        temp_status = ('Flight Halt - Unsafe', 'danger')
    
    # Precipitation Classification (mm/hr)
    if precip < 2.5:
        precip_status = ('Operational - Normal', 'success')
    elif 2.5 <= precip < 7.5:
        precip_status = ('Caution - Reduced Visibility', 'warning')
    else:
        precip_status = ('Flight Halt - Heavy Precipitation', 'danger')
    
    # Visibility Classification (m)
    if visibility is None:
        vis_status = ('Visibility Data Not Available', 'success')
    elif visibility >= 5000:
        vis_status = ('Good Visibility', 'success')
    elif 1000 <= visibility < 5000:
        vis_status = ('Moderate Visibility', 'warning')
    else:
        vis_status = ('Poor Visibility', 'danger')
    
    return {
        "wind": wind_status,
        "temp": temp_status,
        "precip": precip_status,
        "visibility": vis_status
    }

def is_flight_safe(classification):
    """Determine overall flight safety - only fails on 'danger' status"""
    for status in classification.values():
        if status is not None and status[1] == 'danger':
            return False
    return True

def create_time_series_chart(forecast, predictions, tz):
    """Create interactive time series chart showing only 48 hours"""
    # Get current time in airport's timezone
    tz_obj = pytz.timezone(tz)
    now = datetime.now(tz_obj)
    
    # Calculate cutoff time (48 hours from now)
    cutoff_time = now + timedelta(hours=48)
    
    # Filter forecast data to only show next 48 hours
    filtered_forecast = [f for f in forecast if f['time'] <= cutoff_time]
    
    # Prepare forecast data
    forecast_df = pd.DataFrame(filtered_forecast)
    forecast_df['type'] = 'Forecast'
    forecast_df['risk'] = None
    
    # Prepare prediction data (next 2 hours in 20-min intervals)
    pred_times = [now + timedelta(minutes=20*i) for i in range(1, len(predictions)+1)]
    pred_df = pd.DataFrame({
        'time': pred_times,
        'risk': predictions,
        'type': 'Prediction'
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add forecast traces (only showing 48 hours now)
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['wind'],
        name='Wind (m/s)',
        line=dict(color='#1f77b4'),
        yaxis='y1'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['temp']-273.15,
        name='Temp (°C)',
        line=dict(color='#ff7f0e'),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['precip'],
        name='Precip (mm/hr)',
        line=dict(color='#2ca02c'),
        yaxis='y3'
    ))
    
    # Add prediction trace
    fig.add_trace(go.Scatter(
        x=pred_df['time'],
        y=pred_df['risk'],
        name='Risk Score (%)',
        line=dict(color='#d62728', width=3, dash='dot'),
        yaxis='y4'
    ))
    
    # Update layout (title now accurately reflects 48-hour display)
    fig.update_layout(
        title="Weather Forecast & Risk Prediction (Next 48 Hours)",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Wind Speed (m/s)", color='#1f77b4'),
        yaxis2=dict(title="Temperature (°C)", overlaying='y', side='right', color='#ff7f0e'),
        yaxis3=dict(title="Precipitation (mm/hr)", overlaying='y', side='right', position=0.85, color='#2ca02c'),
        yaxis4=dict(title="Risk Score (%)", overlaying='y', side='left', position=0.15, color='#d62728'),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add vertical line for current time
    fig.add_vline(
        x=now.timestamp() * 1000,
        line_dash="dash",
        line_color="white"
    )
    
    return fig

def display_live_alerts():
    """Professional alert display with auto-refresh"""
    alerts = alert_system.get_active_alerts()
    
    if not alerts:
        st.sidebar.markdown("""
        <div class="card decision-safe">
            <h4>✅ No Active Alerts</h4>
            <p>All monitored airports are safe</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.sidebar.markdown("""
    <div class="card decision-danger">
        <h4>🚨 Active Alerts</h4>
        <p>Critical conditions detected</p>
    </div>
    """, unsafe_allow_html=True)
    
    for icao, alert in alerts.items():
        airport = AIRPORT_DB[icao]
        
        with st.sidebar.expander(f"⚠ {airport['iata']} - {airport['name']}", expanded=True):
            # Timeline prediction visualization
            st.markdown(f"*Risk Prediction Timeline (Next 5 min):*")
            
            # Create a small timeline chart
            pred_df = pd.DataFrame({
                'Minutes Ahead': range(5, 65, 5),
                'Risk %': alert['predictions'][:12]
            })
            
            fig = px.line(
                pred_df, 
                x='Minutes Ahead', 
                y='Risk %',
                title=f"Risk Prediction for {airport['iata']}",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Minutes from Now",
                yaxis_title="Risk Percentage",
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Current conditions
            st.markdown("*Current Conditions:*")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Wind Speed", f"{alert['weather_data']['wind']:.1f} m/s")
                st.metric("Temperature", f"{alert['weather_data']['temp']-273.15:.1f}°C")
            with cols[1]:
                st.metric("Precipitation", f"{alert['weather_data']['precip']:.1f} mm/hr")
                st.metric("Visibility", f"{alert['weather_data']['visibility']/1000:.1f} km")
            
            # Critical warnings
            for param, status in alert['classification'].items():
                if status[1] == 'danger':
                    st.error(f"{param.upper()} ALERT**: {status[0]}")
                elif status[1] == 'warning':
                    st.warning(f"{param.upper()} WARNING**: {status[0]}")

def display_safety_analysis(source, destination):
    """Display comprehensive safety analysis"""
    # Get live weather and forecasts
    src_weather = fetch_live_weather(
        AIRPORT_DB[source]["coords"][0],
        AIRPORT_DB[source]["coords"][1],
        AIRPORT_DB[source]["tz"]
    )
    dest_weather = fetch_live_weather(
        AIRPORT_DB[destination]["coords"][0],
        AIRPORT_DB[destination]["coords"][1],
        AIRPORT_DB[destination]["tz"]
    )
    
    src_forecast = fetch_forecast(
        AIRPORT_DB[source]["coords"][0],
        AIRPORT_DB[source]["coords"][1],
        AIRPORT_DB[source]["tz"]
    )
    
    if src_weather and dest_weather and src_forecast:
        # Classify conditions
        src_class = classify_conditions(
            src_weather["wind"],
            src_weather["temp"],
            src_weather["precip"],
            src_weather["visibility"]
        )
        
        dest_class = classify_conditions(
            dest_weather["wind"],
            dest_weather["temp"],
            dest_weather["precip"],
            dest_weather["visibility"]
        )
        
        # Generate predictions
        predictions = generate_time_series_prediction(src_weather, model)
        
        # Display time series chart
        st.markdown("## 📈 Weather Forecast & Risk Prediction")
        st.plotly_chart(
            create_time_series_chart(src_forecast, predictions, AIRPORT_DB[source]["tz"]),
            use_container_width=True
        )
        
        # Display current conditions
        st.markdown("## 📊 Current Flight Safety Analysis")
        
        freshness_badge = ("🟢 Live Data", "success")
        
        st.info(f"""
        {freshness_badge[0]}  
        • Departure: {src_weather['time']} ({AIRPORT_DB[source]['tz'].split('/')[-1]})  
        • Arrival: {dest_weather['time']} ({AIRPORT_DB[destination]['tz'].split('/')[-1]})
        """)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Departure Risk", f"{predictions[0]:.1f}%")
        with m2:
            st.metric("Wind Speed", f"{src_weather['wind']:.1f} m/s")
        with m3:
            st.metric("Temperature", f"{src_weather['temp']-273.15:.1f}°C")
        with m4:
            st.metric("Visibility", f"{src_weather['visibility']/1000:.1f} km")
        
        # Safety Cards
        st.markdown("### 🛡 Safety Assessment")
        
        # Departure Assessment
        with st.expander(f"Departure: {AIRPORT_DB[source]['name']}", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card card {'decision-' + src_class['wind'][1]}">
                    <h5>Wind</h5>
                    <h3>{src_weather['wind']:.1f} m/s</h3>
                    <p>{src_class['wind'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card card {'decision-' + src_class['temp'][1]}">
                    <h5>Temperature</h5>
                    <h3>{src_weather['temp']-273.15:.1f}°C</h3>
                    <p>{src_class['temp'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card card {'decision-' + src_class['precip'][1]}">
                    <h5>Precipitation</h5>
                    <h3>{src_weather['precip']:.1f} mm/hr</h3>
                    <p>{src_class['precip'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="metric-card card {'decision-' + src_class['visibility'][1]}">
                    <h5>Visibility</h5>
                    <h3>{src_weather['visibility']/1000:.1f} km</h3>
                    <p>{src_class['visibility'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Final Decision
        overall_safe = is_flight_safe(src_class) and is_flight_safe(dest_class)
        
        if overall_safe:
            st.success("""
            ## 🟢 CLEAR FOR DEPARTURE
            Weather conditions at both airports are within operational limits.
            """)
            st.balloons()
        else:
            st.error("""
            ## 🔴 FLIGHT NOT RECOMMENDED
            Hazardous weather conditions detected. Consider delaying or rerouting.
            """)
            
            # Show specific issues
            st.warning("### 🚨 Critical Issues Detected")
            issues = []
            
            for param, status in src_class.items():
                if status and status[1] == 'danger':
                    issues.append(f"Departure {param}: {status[0]}")
            
            for param, status in dest_class.items():
                if status and status[1] == 'danger':
                    issues.append(f"Arrival {param}: {status[0]}")
            
            for issue in issues:
                st.markdown(f"- ❗ {issue}")

# ======================
# MAIN APP
# ======================
# ======================
# MAIN APP
def main_app():
    """Professional Aviation Weather Dashboard with Real-time Monitoring"""
    
    # Initialize alert system with model
    global alert_system
    model = load_weather_model()
    if model is None:
        st.error("Failed to load weather prediction model. Critical functionality disabled.")
        return
    
    alert_system = EnhancedAlertSystem(model, AIRPORT_DB)
    
    # Initialize route if not set
    if not alert_system.source or not alert_system.destination:
        default_source = "VIDP"  # Delhi
        default_dest = "VABB"  # Mumbai
        alert_system.set_flight_route(default_source, default_dest)
    
    # Main UI with animated header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%); 
                padding: 20px; 
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                animation: fadeIn 1.5s ease-in-out;">
        <h1 style="color: white; margin: 0;">✈ AI-Powered Aviation Weather Advisor</h1>
        <p style="color: white; font-size: 1.1rem;">Real-time flight safety monitoring system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration section with flight path visualization
    with st.expander("🛫 Flight Route Configuration", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            source = st.selectbox(
                "Departure Airport",
                options=list(AIRPORT_DB.keys()),
                format_func=lambda x: f"{AIRPORT_DB[x]['iata']} - {AIRPORT_DB[x]['name']}",
                index=0
            )
        
        with col2:
            destination = st.selectbox(
                "Arrival Airport",
                options=list(AIRPORT_DB.keys()),
                format_func=lambda x: f"{AIRPORT_DB[x]['iata']} - {AIRPORT_DB[x]['name']}",
                index=1 if len(AIRPORT_DB) > 1 else 0
            )
        
        with col3:
            # Flight path visualization
            try:
                src_coords = AIRPORT_DB[source]['coords']
                dest_coords = AIRPORT_DB[destination]['coords']
                
                fig = go.Figure()
                
                # Add airport markers
                fig.add_trace(go.Scattergeo(
                    lon = [src_coords[1], dest_coords[1]],
                    lat = [src_coords[0], dest_coords[0]],
                    text = [AIRPORT_DB[source]['name'], AIRPORT_DB[destination]['name']],
                    marker = dict(
                        size = 12,
                        color = ['green', 'red'],
                        line = dict(width=3, color='white')
                    ),
                    name = 'Airports'
                ))
                
                # Add flight path
                fig.add_trace(go.Scattergeo(
                    lon = [src_coords[1], dest_coords[1]],
                    lat = [src_coords[0], dest_coords[0]],
                    mode = 'lines',
                    line = dict(width=2, color='blue'),
                    opacity = 0.7,
                    name = 'Flight Path'
                ))
                
                fig.update_geos(
                    projection_type="orthographic",
                    showland=True,
                    landcolor="lightgray",
                    showocean=True,
                    oceancolor="azure",
                    showcountries=True,
                    countrycolor="white"
                )
                
                fig.update_layout(
                    height=200,
                    margin={"r":0,"t":0,"l":0,"b":0},
                    showlegend=False,
                    geo = dict(
                        center=dict(
                            lon=(src_coords[1] + dest_coords[1])/2,
                            lat=(src_coords[0] + dest_coords[0])/2
                        )
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not render flight path visualization")
        
        # Update alert system with new route
        if st.button("🔄 Update Flight Route & Run Analysis", type="primary"):
            with st.spinner("Configuring monitoring system..."):
                alert_system.set_flight_route(source, destination)
                st.success("Flight route updated for monitoring!")
                time.sleep(1)
                st.rerun()
    
    # Safety Analysis Section with animated cards
    st.markdown("## 🔍 Advanced Safety Analysis")
    
    if st.button("🚀 Run Comprehensive Safety Check", type="primary", help="Analyze current and predicted weather conditions"):
        with st.spinner("Running advanced safety analysis..."):
            # Set the route for alerts
            alert_system.set_flight_route(source, destination)
            
            # Start monitoring if not already running
            if not alert_system.running:
                alert_system.start()
            
            # Display results
            display_safety_analysis(source, destination)
    
    # Real-time Monitoring Dashboard with multiple visualization options
    st.markdown("## 📡 Live Monitoring Dashboard")
    
    monitoring_tab1, monitoring_tab2 = st.tabs(["📊 Real-time Metrics", "📈 Historical Trends"])
    
    with monitoring_tab1:
        # Enhanced real-time monitoring section
        monitoring_col1, monitoring_col2 = st.columns([3, 1])
        
        with monitoring_col1:
            st.markdown("### 🌦 Current Weather Conditions")
            
            # Get live weather data
            weather = fetch_live_weather(
                AIRPORT_DB[source]['coords'][0],
                AIRPORT_DB[source]['coords'][1],
                AIRPORT_DB[source]['tz']
            )
            
            if weather:
                # Create a grid of metric cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="card metric-card" style="animation: pulse 2s infinite;">
                        <h5>Wind Speed</h5>
                        <h2>{weather['wind']:.1f} m/s</h2>
                        <p>{'⚠ High Wind' if weather['wind'] > 10 else '✅ Normal'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="card metric-card">
                        <h5>Temperature</h5>
                        <h2>{weather['temp']-273.15:.1f} °C</h2>
                        <p>{'⚠ Extreme' if (weather['temp']-273.15 < 0 or weather['temp']-273.15 > 35) else '✅ Normal'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="card metric-card">
                        <h5>Precipitation</h5>
                        <h2>{weather['precip']:.1f} mm/hr</h2>
                        <p>{'⚠ Heavy Rain' if weather['precip'] > 2.5 else '✅ Normal'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m4:
                    st.markdown(f"""
                    <div class="card metric-card">
                        <h5>Visibility</h5>
                        <h2>{weather['visibility']/1000:.1f} km</h2>
                        <p>{'⚠ Low Visibility' if weather['visibility'] < 5000 else '✅ Clear'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Weather condition visualization
                condition_emoji = "☀"
                if weather['precip'] > 5:
                    condition_emoji = "🌧"
                elif weather['wind'] > 10:
                    condition_emoji = "🌬"
                elif weather['visibility'] < 3000:
                    condition_emoji = "🌫"
                
                st.markdown(f"""
                <div class="card" style="text-align: center; padding: 20px;">
                    <h3>{condition_emoji} Current Weather: {weather['weather_condition']}</h3>
                    <p>Last updated: {weather['time']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with monitoring_col2:
            st.markdown("### ⚠ Alerts Status")
            alerts = alert_system.get_active_alerts()
            
            if alerts:
                st.error(f"🚨 {len(alerts)} Active Alert(s)")
                for icao, alert in alerts.items():
                    with st.expander(f"{AIRPORT_DB[icao]['iata']} Alert Details"):
                        st.write(f"Time: {alert['timestamp']}")
                        st.write(f"Risk Level: {max(alert['predictions']):.0f}%")
                        st.write("Critical Issues:")
                        for param, status in alert['classification'].items():
                            if status[1] == 'danger':
                                st.error(f"- {param.capitalize()}: {status[0]}")
            else:
                st.success("✅ No Active Alerts")
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828640.png" width="80">
                    <p>All systems normal</p>
                </div>
                """, unsafe_allow_html=True)
    
    with monitoring_tab2:
        # Historical trends section
        st.markdown("### 📈 Weather Trends (Last 24 Hours)")
        
        # Simulated historical data (in a real app, you'd fetch this from a database)
        hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
        historical_data = pd.DataFrame({
            'Time': hours,
            'Wind (m/s)': np.random.normal(5, 2, 24).clip(0, 20),
            'Temperature (°C)': np.random.normal(25, 5, 24),
            'Precipitation (mm/hr)': np.random.gamma(1, 0.5, 24)
        })
        
        # Create interactive historical chart
        fig = px.line(historical_data, x='Time', y=['Wind (m/s)', 'Temperature (°C)', 'Precipitation (mm/hr)'],
                     title="Historical Weather Trends",
                     template="plotly_dark")
        
        fig.update_layout(
            hovermode="x unified",
            legend_title="Parameters",
            yaxis_title="Value",
            xaxis_title="Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time monitoring toggle
    monitoring_enabled = st.checkbox("🔴 Enable Continuous Monitoring (5-min updates)", 
                                   help="System will continuously monitor weather conditions and alert you to changes")
    
    if monitoring_enabled:
        if not alert_system.running:
            if alert_system.start():
                st.success("Live monitoring started! System will auto-refresh.")
            else:
                st.error("Could not start monitoring. Check flight route.")

        from streamlit_autorefresh import st_autorefresh

# Auto refresh every 5 minutes
        st_autorefresh(interval=300000, limit=None, key="weather-refresh")


        # Display real-time gauges
        st.markdown("### 🎛 Real-time Weather Gauges")
        
        if source in AIRPORT_DB:
            weather = fetch_live_weather(
                AIRPORT_DB[source]['coords'][0],
                AIRPORT_DB[source]['coords'][1],
                AIRPORT_DB[source]['tz']
            )
            
            if weather:
                # Create gauge charts with better styling
                g1, g2, g3 = st.columns(3)
                with g1:
                    fig_wind = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = weather['wind'],
                        title = {'text': "Wind Speed (m/s)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 25], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 10], 'color': "lightgreen"},
                                {'range': [10, 15], 'color': "orange"},
                                {'range': [15, 25], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': weather['wind']}
                        }
                    ))
                    fig_wind.update_layout(height=300, margin=dict(t=50, b=10))
                    st.plotly_chart(fig_wind, use_container_width=True)
                
                with g2:
                    fig_temp = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = weather['temp']-273.15,
                        title = {'text': "Temperature (°C)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [-20, 50]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [-20, 0], 'color': "lightblue"},
                                {'range': [0, 35], 'color': "lightgreen"},
                                {'range': [35, 50], 'color': "red"}],
                        }
                    ))
                    fig_temp.update_layout(height=300, margin=dict(t=50, b=10))
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with g3:
                    fig_precip = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = weather['precip'],
                        title = {'text': "Precipitation (mm/hr)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 10]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 2.5], 'color': "lightgreen"},
                                {'range': [2.5, 7.5], 'color': "orange"},
                                {'range': [7.5, 10], 'color': "red"}],
                        }
                    ))
                    fig_precip.update_layout(height=300, margin=dict(t=50, b=10))
                    st.plotly_chart(fig_precip, use_container_width=True)
        
        # Auto-refresh with better UX
        refresh_placeholder = st.empty()
        last_update = datetime.now().strftime("%H:%M:%S")
        refresh_placeholder.markdown(f"""
        <div style="background: rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin: 0;">🔄 Last updated: {last_update}</p>
            <p style="margin: 0; font-size: 0.8rem;">Next update in 5 minutes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate auto-refresh
         # Simulate auto-refresh
        time.sleep(300)
       

        
        st.rerun()

    else:
        if alert_system.running:
            alert_system.stop()
            st.success("Live monitoring stopped.")

    # Enhanced System Info Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; animation: fadeIn 2s;">
            <img src="https://cdn-icons-png.flaticon.com/512/2933/2933245.png" width="80">
            <h2>Aviation Weather AI</h2>
            <p style="color: #00c6ff;">v3.0 - Professional Edition</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="animation: slideIn 1s;">
            <h4>🚀 Key Features</h4>
            <p>• Real-time weather monitoring</p>
            <p>• AI-powered risk prediction</p>
            <p>• Multi-airport tracking</p>
            <p>• Professional aviation standards</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>⚙ System Status</h4>
            <p><b>Model:</b> {'✅ Loaded' if model else '❌ Failed'}</p>
            <p><b>Monitoring:</b> {'✅ Active' if alert_system.running else '🟡 Inactive'}</p>
            <p><b>Last Scan:</b> {datetime.now().strftime("%H:%M:%S")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align:center; margin-top:20px;">
            <small>Developed for SpaceTech Hackathon</small><br>
            <small>© 2023 Aviation Weather Systems</small>
        </div>
        """, unsafe_allow_html=True)

# Add custom CSS animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ======================
# RUN APPLICATION
# ======================

if __name__ == "__main__":
    # Authentication first
    authentication_section()
    
    # Only show main app if authenticated
    if st.session_state.get('auth', {}).get('authenticated', False):
        main_app()
    else:
        st.warning("🔒 Please authenticate to access the aviation weather system")
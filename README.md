# AI-Powered Aviation Weather Advisor

## Overview

This is a professional-grade aviation weather monitoring system that leverages artificial intelligence to provide real-time flight safety analysis and predictions. It integrates live weather data with a pre-trained machine learning model to assess potential hazards at departure and arrival airports. The system offers user authentication, route configuration, comprehensive safety checks, live monitoring dashboards, and alert notifications.

## Key Features

* *Secure User Authentication:* Robust registration and login system with persistent user data storage.
* *Flight Route Configuration:* Users can select departure and arrival airports from a comprehensive database.
* *Real-time Weather Monitoring:* Fetches and displays current weather conditions for selected airports.
* *AI-Powered Risk Prediction:* Utilizes a pre-trained LSTM model to predict short-term weather risks.
* *Comprehensive Safety Analysis:* Assesses wind speed, temperature, precipitation, and visibility against aviation safety standards.
* *Live Alert System:* Continuously monitors weather conditions and triggers alerts for critical situations.
* *Interactive Visualizations:* Presents weather data and risk predictions through dynamic charts and gauges.
* *Historical Weather Trends:* Displays historical weather data for trend analysis.
* *Automatic Updates:* Real-time data and alert updates at regular intervals.
* *Professional User Interface:* Clean, intuitive, and responsive design optimized for aviation professionals.

## Installation

1.  *Clone the repository* (if applicable).
2.  *Ensure Python 3.6 or higher is installed.*
3.  *Install the required Python libraries:*
    bash
    pip install -r requirements.txt
    
    *(Note: A requirements.txt file listing all dependencies would be needed for this step to be fully functional).*
4.  *Obtain an OpenWeatherMap API key.* Sign up for a free API key at [https://openweathermap.org/api](https://openweathermap.org/api).
5.  *Store the API key as a Streamlit secret.* Create a .streamlit/secrets.toml file in your project directory (or configure Streamlit secrets through your deployment platform) with the following content:
    toml
    OPENWEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
    
    Replace "YOUR_OPENWEATHERMAP_API_KEY" with your actual API key.
6.  **Ensure the pre-trained Keras model (weather_prediction_lstm.keras) is in the same directory as the main script.** (Note: The model file is not included in the provided code, so you would need to obtain or train this model separately).

## Usage

1.  *Run the Streamlit application:*
    bash
    streamlit run dost.py
    
    Replace your_script_name.py with the name of your Python script containing the code.
2.  *Authenticate:* On the sidebar, either log in with an existing account or register a new user.
3.  *Configure Flight Route:* In the "Flight Route Configuration" section, select the departure and arrival airports from the dropdown menus.
4.  *Run Safety Analysis:* Click the "Run Comprehensive Safety Check" button to analyze the current and predicted weather conditions for the selected route.
5.  *Monitor Live Conditions:* The "Live Monitoring Dashboard" section displays real-time weather metrics, alerts, and historical trends.
6.  *Enable Continuous Monitoring:* Check the "Enable Continuous Monitoring" box to activate automatic weather updates and alerts every 5 minutes.

![Screenshot 2025-05-06 192346](https://github.com/user-attachments/assets/be1bf8e7-6f77-4c7a-89a5-2e5c13800982)

![Screenshot 2025-05-06 192409](https://github.com/user-attachments/assets/d0d1b573-bce7-470b-97f6-52fe28233ce3)

![Screenshot 2025-05-06 192427](https://github.com/user-attachments/assets/7d97cc4e-2c77-4976-9004-e8019f7bba16)

![Screenshot 2025-05-06 192450](https://github.com/user-attachments/assets/e3f3c6e1-af02-49e8-a014-4e5028def8af)

![Screenshot 2025-05-06 192510](https://github.com/user-attachments/assets/1e078471-622b-4eeb-8810-9f9833651a71)






## Code Structure

The code is organized into several key sections:

* *Standard and Third-Party Libraries:* Imports necessary modules for functionality.
* **Enhanced Authentication System (AuthSystem class):** Manages user registration, login, and persistent storage using users.json.
* **Enhanced Alert System (EnhancedAlertSystem class):** Monitors weather conditions for selected airports and triggers alerts based on defined thresholds and AI predictions.
* *Configuration:* Sets up the Streamlit page, API base URL, and defines custom CSS for styling.
* *Constants & Data:* Contains the AIRPORT_DB, a dictionary of airport ICAO codes, names, coordinates, IATA codes, and timezones.
* *Cached Resources:* Defines cached functions (load_weather_model, fetch_forecast, fetch_live_weather) to efficiently load the ML model and fetch weather data from the OpenWeatherMap API.
* *Utility Functions:* Includes functions for generating time series predictions, classifying weather conditions, determining flight safety, and creating interactive charts.
* *Display Functions:* Functions to display live alerts, safety analysis, and the main application interface.
* **Main Application (main_app function):** Orchestrates the different sections of the application, including route configuration, safety analysis, and live monitoring.
* **Authentication Section (authentication_section function):** Handles user login and registration UI.
* **Main Execution (if __name__ == "__main__":): Runs the authentication system first and then the main application if the user is authenticated.

## Dependencies

* streamlit
* streamlit_autorefresh
* numpy
* pandas
* plotly.express
* plotly.graph_objects
* requests
* pytz
* keras
* scikit-learn
* tensorflow

## Pre-trained Model

This application relies on a pre-trained Keras LSTM model (weather_prediction_lstm.keras) for weather risk prediction. This model is not included in the code and needs to be provided separately. It is expected to be trained on relevant weather data to predict a risk score based on current weather conditions.

## API Key

The application uses the OpenWeatherMap API to fetch live weather data and forecasts. You need to obtain your own API key and configure it as a Streamlit secret as described in the Installation section.

## Contributing

Contributions to this project are welcome. Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure they are well-documented.
4.  Write tests for your changes.
5.  Submit a pull request with a clear description of your changes.

## License

This project is licensed under the [Specify License Here] License.

## Acknowledgments

* This project utilizes the [OpenWeatherMap API](https://openweathermap.org/api) for weather data.
* The pre-trained machine learning model was developed using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/).
* The user interface is built with the [Streamlit](https://streamlit.io/) framework.
* The interactive visualizations are created using the [Plotly](https://plotly.com/) library.
* The auto-refresh functionality is provided by the [streamlit-autorefresh](https://github.com/tvst/streamlit-autorefresh) library.
*

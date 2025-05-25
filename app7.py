import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import json
import time
import math
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')

# Default Grok API Key (provided)
DEFAULT_GROQ_API_KEY = "Your API"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"

class PollutionAnalyzer:
    def __init__(self, weather_api_key, groq_api_key):
        self.weather_api_key = weather_api_key
        self.groq_api_key = groq_api_key if groq_api_key else DEFAULT_GROQ_API_KEY
        self.weather_base_url = WEATHER_BASE_URL
        self.geolocator = Nominatim(user_agent="pollution_analyzer")
        
    def validate_api_keys(self):
        """Validate that API keys are provided"""
        if not self.weather_api_key:
            st.error("Please provide a valid WeatherAPI key.")
            return False
        if not self.groq_api_key:
            st.error("Please provide a valid Grok API key or use the default.")
            return False
        return True
    
    def get_air_quality_data(self, location):
        """Get current air quality data for a location"""
        if not self.validate_api_keys():
            return None
        try:
            url = f"{self.weather_base_url}/current.json"
            params = {
                'key': self.weather_api_key,
                'q': location,
                'aqi': 'yes'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching air quality data: {e}")
            return None
    
    def get_weather_forecast(self, location, days=14):
        """Get weather and air quality forecast for up to 14 days"""
        if not self.validate_api_keys():
            return None
        try:
            url = f"{self.weather_base_url}/forecast.json"
            params = {
                'key': self.weather_api_key,
                'q': location,
                'days': days,
                'aqi': 'yes'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching forecast: {e}")
            return None
    
    def process_forecast_data(self, forecast_data, days):
        """Process forecast data to get daily pollutant averages"""
        if not forecast_data or 'forecast' not in forecast_data:
            return None
        
        daily_data = []
        pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
        for day in forecast_data['forecast']['forecastday'][:days]:
            date = day['date']
            daily_avg = {'date': date}
            for pollutant in pollutants:
                values = [hour['air_quality'].get(pollutant, 0) for hour in day.get('hour', []) if hour['air_quality'].get(pollutant)]
                daily_avg[pollutant] = np.mean(values) if values else 0
            daily_data.append(daily_avg)
        
        return pd.DataFrame(daily_data)
    
    def analyze_with_ai(self, data, context):
        """Use Grok API for AI analysis"""
        if not self.validate_api_keys():
            return None
        try:
            headers = {
                'Authorization': f'Bearer {self.groq_api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""
            **Context**: {context}
            **Data**: 
            ```json
            {json.dumps(data, indent=2)}
            ```
            
            Provide a detailed analysis and actionable recommendations tailored to the context. Ensure recommendations are practical, specific, and focused on Indian conditions where applicable. Structure the response clearly with headings and bullet points for readability.
            """
            
            payload = {
                'model': 'llama3-70b-8192',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 1500
            }
            
            response = requests.post(GROQ_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"AI analysis unavailable: {e}"

class AgricultureModule:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.crop_sensitivity = {
            'wheat': {'o3': 0.12, 'so2': 0.08, 'no2': 0.05, 'pm25': 0.10},
            'rice': {'o3': 0.05, 'so2': 0.03, 'no2': 0.02, 'pm25': 0.04},
            'corn': {'o3': 0.15, 'so2': 0.10, 'no2': 0.07, 'pm25': 0.12},
            'soybean': {'o3': 0.18, 'so2': 0.12, 'no2': 0.08, 'pm25': 0.14},
            'cotton': {'o3': 0.10, 'so2': 0.06, 'no2': 0.04, 'pm25': 0.08}
        }
    
    def predict_crop_impact(self, location, crop_type):
        """Predict pollution impact on crops based on current data"""
        air_data = self.analyzer.get_air_quality_data(location)
        if not air_data or 'current' not in air_data:
            return None
        
        air_quality = air_data['current'].get('air_quality', {})
        sensitivity = self.crop_sensitivity.get(crop_type.lower(), self.crop_sensitivity['wheat'])
        
        yield_loss = 0
        pollutant_impacts = {}
        
        for pollutant, coefficient in sensitivity.items():
            if pollutant == 'pm25':
                value = air_quality.get('pm2_5', 0)
                normalized = min(value / 15, 3.0)
            elif pollutant == 'o3':
                value = air_quality.get('o3', 0)
                normalized = min(value / 100, 3.0)
            elif pollutant == 'so2':
                value = air_quality.get('so2', 0)
                normalized = min(value / 20, 3.0)
            elif pollutant == 'no2':
                value = air_quality.get('no2', 0)
                normalized = min(value / 40, 3.0)
            else:
                continue
                
            impact = coefficient * normalized * 100
            pollutant_impacts[pollutant] = {
                'value': value,
                'impact_percent': impact,
                'risk_level': self.get_risk_level(impact)
            }
            yield_loss += impact
        
        return {
            'total_yield_loss': min(yield_loss, 50),
            'pollutant_impacts': pollutant_impacts,
            'air_quality_data': air_quality,
            'location_data': air_data['location']
        }
    
    def get_risk_level(self, impact):
        """Determine risk level based on impact percentage"""
        if impact < 2: return "Low"
        elif impact < 5: return "Moderate"
        elif impact < 10: return "High"
        else: return "Critical"
    
    def generate_farming_recommendations(self, impact_data, forecast_data, crop_type, forecast_days):
        """Generate AI-powered farming recommendations"""
        combined_data = {
            'current_impact': impact_data,
            'forecast_data': forecast_data.to_dict('records') if forecast_data is not None else []
        }
        context = f"""
        Agriculture AI-GIS Analysis for {crop_type} farming in India over the next {forecast_days} days:
        Provide recommendations based on current and forecasted air quality data for:
        - Precision agriculture strategies
        - Fertilizer application adjustments
        - Crop protection measures
        - Optimal planting/harvesting timing
        - Pollutant-resistant varieties
        - Soil management practices
        Focus on practical, cost-effective solutions tailored for Indian farming conditions.
        """
        return self.analyzer.analyze_with_ai(combined_data, context)

class SmartCitiesModule:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.indian_cities = {
            'Delhi': (28.6139, 77.2090), 'Mumbai': (19.0760, 72.8777), 'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }
    
    def create_pollution_map(self, cities_data):
        """Create interactive pollution map"""
        center_lat, center_lon = 20.5937, 78.9629
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        for city, data in cities_data.items():
            if data and 'current' in data:
                air_quality = data['current'].get('air_quality', {})
                location_info = data['location']
                pm25 = air_quality.get('pm2_5', 0)
                
                color = 'green' if pm25 <= 12 else 'yellow' if pm25 <= 35 else 'orange' if pm25 <= 55 else 'red'
                
                popup_text = f"""
                <b>{city}</b><br>
                PM2.5: {pm25} μg/m³<br>
                PM10: {air_quality.get('pm10', 'N/A')} μg/m³<br>
                O3: {air_quality.get('o3', 'N/A')} μg/m³<br>
                NO2: {air_quality.get('no2', 'N/A')} μg/m³<br>
                SO2: {air_quality.get('so2', 'N/A')} μg/m³<br>
                CO: {air_quality.get('co', 'N/A')} μg/m³
                """
                
                folium.CircleMarker(
                    location=[location_info['lat'], location_info['lon']],
                    radius=10 + (pm25 / 10),
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        return m
    
    def predict_air_quality_trends(self, forecast_data, days):
        """Predict air quality trends using forecast data"""
        if not forecast_data or 'forecast' not in forecast_data:
            return None
        
        daily_predictions = []
        for day in forecast_data['forecast']['forecastday'][:days]:
            for hour in day.get('hour', []):
                air_quality = hour.get('air_quality', {})
                daily_predictions.append({
                    'datetime': hour['time'],
                    'temp': hour['temp_c'],
                    'humidity': hour['humidity'],
                    'wind_speed': hour['wind_kph'],
                    'pressure': hour['pressure_mb'],
                    'predicted_pm25': air_quality.get('pm2_5', self.predict_pm25(hour)),
                    'predicted_aqi': self.calculate_aqi(air_quality.get('pm2_5', self.predict_pm25(hour)))
                })
        return daily_predictions
    
    def predict_pm25(self, weather_data):
        """Simple PM2.5 prediction model"""
        base_pm25 = 35
        temp_factor = max(0, (weather_data['temp_c'] - 25) / 10) * 5
        humidity_factor = (weather_data['humidity'] - 50) / 50 * 10
        wind_factor = max(0, (10 - weather_data['wind_kph']) / 10) * 15
        predicted_pm25 = base_pm25 + temp_factor + humidity_factor + wind_factor
        return max(5, min(predicted_pm25, 200))
    
    def calculate_aqi(self, pm25):
        """Calculate AQI from PM2.5"""
        if pm25 <= 12: return int((50 / 12) * pm25)
        elif pm25 <= 35.4: return int(50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1))
        elif pm25 <= 55.4: return int(100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5))
        elif pm25 <= 150.4: return int(150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5))
        else: return min(300, int(200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)))
    
    def generate_city_recommendations(self, comparison_data, forecast_data, forecast_days):
        """Generate AI-powered city management recommendations"""
        combined_data = {
            'current_data': comparison_data,
            'forecast_data': forecast_data.to_dict('records') if forecast_data is not None else []
        }
        context = f"""
        Smart Cities Air Quality Management Analysis in India over the next {forecast_days} days:
        Provide recommendations based on current and forecasted air quality data for:
        - Traffic management strategies
        - Industrial emission controls
        - Public transportation optimization
        - Green infrastructure development
        - Emergency response protocols
        - Citizen health advisories
        Focus on actionable solutions for Indian urban environments.
        """
        return self.analyzer.analyze_with_ai(combined_data, context)

class HealthcareModule:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.risk_profiles = {
            'child': {'pm25': 1.5, 'o3': 1.3, 'no2': 1.2, 'so2': 1.4},
            'adult': {'pm25': 1.0, 'o3': 1.0, 'no2': 1.0, 'so2': 1.0},
            'elderly': {'pm25': 1.8, 'o3': 1.6, 'no2': 1.4, 'so2': 1.7},
            'asthma': {'pm25': 2.2, 'o3': 2.0, 'no2': 1.8, 'so2': 2.1},
            'heart_disease': {'pm25': 2.0, 'o3': 1.7, 'no2': 1.5, 'so2': 1.8}
        }
    
    def assess_health_risk(self, location, age_group, conditions=None):
        """Assess personalized health risk based on current air quality"""
        air_data = self.analyzer.get_air_quality_data(location)
        if not air_data or 'current' not in air_data:
            return None
        
        air_quality = air_data['current'].get('air_quality', {})
        profile = (self.risk_profiles['asthma'] if conditions and 'asthma' in conditions else
                   self.risk_profiles['heart_disease'] if conditions and 'heart_disease' in conditions else
                   self.risk_profiles.get(age_group, self.risk_profiles['adult']))
        
        health_risks = {}
        total_risk_score = 0
        pollutants = {'pm25': air_quality.get('pm2_5', 0), 'o3': air_quality.get('o3', 0),
                      'no2': air_quality.get('no2', 0), 'so2': air_quality.get('so2', 0)}
        
        for pollutant, value in pollutants.items():
            if pollutant in profile:
                normalized = (value / 15 if pollutant == 'pm25' else value / 100 if pollutant == 'o3' else
                             value / 40 if pollutant == 'no2' else value / 20)
                risk_score = min(normalized * profile[pollutant] * 10, 10)
                total_risk_score += risk_score
                health_risks[pollutant] = {
                    'value': value,
                    'risk_score': risk_score,
                    'risk_level': self.get_health_risk_level(risk_score)
                }
        
        overall_risk = min(total_risk_score / 4, 10)
        return {
            'overall_risk_score': overall_risk,
            'overall_risk_level': self.get_health_risk_level(overall_risk),
            'pollutant_risks': health_risks,
            'recommendations': self.generate_health_recommendations(overall_risk, age_group, conditions),
            'air_quality_data': air_quality
        }
    
    def get_health_risk_level(self, score):
        """Determine health risk level"""
        if score < 2: return "Low"
        elif score < 4: return "Moderate"
        elif score < 6: return "High"
        elif score < 8: return "Very High"
        else: return "Hazardous"
    
    def generate_health_recommendations(self, risk_score, age_group, conditions):
        """Generate basic health recommendations"""
        recommendations = []
        if risk_score < 2:
            recommendations.extend(["Air quality is good. Normal outdoor activities are safe.",
                                   "Continue regular exercise routines."])
        elif risk_score < 4:
            recommendations.extend(["Moderate air pollution. Sensitive individuals should limit outdoor activities.",
                                   "Consider indoor exercise on high pollution days."])
        elif risk_score < 6:
            recommendations.extend(["Unhealthy air quality. Limit outdoor activities, especially vigorous exercise.",
                                   "Use air purifiers indoors.", "Wear N95 masks when outdoors."])
        else:
            recommendations.extend(["Hazardous air quality. Avoid outdoor activities.",
                                   "Stay indoors with air purification.",
                                   "Seek medical attention if experiencing respiratory issues."])
        
        if age_group == 'child':
            recommendations.append("Keep children indoors during high pollution periods.")
        elif age_group == 'elderly':
            recommendations.append("Elderly individuals should be extra cautious and monitor symptoms.")
        if conditions:
            if 'asthma' in conditions:
                recommendations.append("Keep rescue inhalers readily available.")
            if 'heart_disease' in conditions:
                recommendations.append("Monitor heart rate and blood pressure regularly.")
        return recommendations
    
    def generate_ai_health_recommendations(self, risk_assessment, forecast_data, age_group, conditions, forecast_days):
        """Generate AI-powered health recommendations"""
        combined_data = {
            'current_risk': risk_assessment,
            'forecast_data': forecast_data.to_dict('records') if forecast_data is not None else []
        }
        context = f"""
        Healthcare Air Quality Risk Analysis in India over the next {forecast_days} days:
        Patient Profile: {age_group} with conditions: {conditions}
        Current Risk Level: {risk_assessment.get('overall_risk_level', 'Unknown')}
        Provide recommendations based on current and forecasted air quality data for:
        - Daily activity modifications
        - Medication adjustments if needed
        - Protective measures
        - When to seek medical attention
        - Long-term health monitoring
        - Indoor air quality improvements
        Focus on evidence-based medical guidance tailored for Indian conditions.
        """
        return self.analyzer.analyze_with_ai(combined_data, context)

class TravelEcoTourismModule:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.indian_cities = {
            'Delhi': (28.6139, 77.2090), 'Mumbai': (19.0760, 72.8777), 'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }
    
    def optimize_low_pollution_route(self, start_city, end_city):
        """Optimize travel route to minimize pollution exposure"""
        start_data = self.analyzer.get_air_quality_data(start_city)
        end_data = self.analyzer.get_air_quality_data(end_city)
        if not start_data or not end_data:
            return None
        
        start_aqi = self.calculate_aqi(start_data['current']['air_quality'].get('pm2_5', 0))
        end_aqi = self.calculate_aqi(end_data['current']['air_quality'].get('pm2_5', 0))
        
        route_score = (start_aqi + end_aqi) / 2
        route_status = "Low Pollution" if route_score < 50 else "Moderate Pollution" if route_score < 100 else "High Pollution"
        
        return {
            'start_city': start_city,
            'end_city': end_city,
            'start_aqi': start_aqi,
            'end_aqi': end_aqi,
            'route_score': route_score,
            'route_status': route_status,
            'start_location': start_data['location'],
            'end_location': end_data['location']
        }
    
    def identify_clean_air_destinations(self, cities):
        """Identify clean-air destinations for eco-tourism"""
        clean_destinations = []
        for city in cities:
            air_data = self.analyzer.get_air_quality_data(city)
            if air_data and 'current' in air_data:
                aqi = self.calculate_aqi(air_data['current']['air_quality'].get('pm2_5', 0))
                if aqi < 50:
                    clean_destinations.append({
                        'city': city,
                        'aqi': aqi,
                        'location': air_data['location']
                    })
        return clean_destinations
    
    def map_pollution_hotspots(self, cities_data):
        """Map pollution hotspots"""
        center_lat, center_lon = 20.5937, 78.9629
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        for city, data in cities_data.items():
            if data and 'current' in data:
                air_quality = data['current'].get('air_quality', {})
                location_info = data['location']
                pm25 = air_quality.get('pm2_5', 0)
                color = 'red' if pm25 > 55 else 'orange' if pm25 > 35 else 'yellow' if pm25 > 12 else 'green'
                
                popup_text = f"""
                <b>{city}</b><br>
                PM2.5: {pm25} μg/m³<br>
                AQI: {self.calculate_aqi(pm25)}
                """
                folium.CircleMarker(
                    location=[location_info['lat'], location_info['lon']],
                    radius=10 + (pm25 / 10),
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        return m
    
    def calculate_aqi(self, pm25):
        """Calculate AQI from PM2.5"""
        if pm25 <= 12: return int((50 / 12) * pm25)
        elif pm25 <= 35.4: return int(50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1))
        elif pm25 <= 55.4: return int(100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5))
        elif pm25 <= 150.4: return int(150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5))
        else: return min(300, int(200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)))
    
    def generate_travel_recommendations(self, travel_data, start_forecast, end_forecast, forecast_days):
        """Generate AI-powered travel recommendations"""
        combined_data = {
            'current_data': travel_data,
            'start_city_forecast': start_forecast.to_dict('records') if start_forecast is not None else [],
            'end_city_forecast': end_forecast.to_dict('records') if end_forecast is not None else []
        }
        context = f"""
        Sustainable Travel and Eco-Tourism Analysis in India over the next {forecast_days} days:
        Air pollution can reduce tourist arrivals by 10-15% in heavily polluted urban areas (UNWTO, 2019).
        Provide recommendations based on current and forecasted air quality data for:
        - Low-pollution travel routes
        - Eco-tourism destination promotion
        - Real-time pollution hotspot avoidance
        - Sustainable travel policies
        - Community-based tourism initiatives
        - Traveler health and safety measures
        Focus on actionable, India-specific solutions that promote sustainable tourism.
        """
        return self.analyzer.analyze_with_ai(combined_data, context)

class RealEstateUrbanPlanningModule:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.indian_cities = {
            'Delhi': (28.6139, 77.2090), 'Mumbai': (19.0760, 72.8777), 'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }
    
    def assess_site_suitability(self, location):
        """Assess site suitability for real estate based on current air quality"""
        air_data = self.analyzer.get_air_quality_data(location)
        if not air_data or 'current' not in air_data:
            return None
        
        air_quality = air_data['current'].get('air_quality', {})
        pm25 = air_quality.get('pm2_5', 0)
        aqi = self.calculate_aqi(pm25)
        
        suitability_score = max(0, 100 - (aqi / 3))
        suitability_level = "High" if suitability_score > 80 else "Moderate" if suitability_score > 50 else "Low"
        
        return {
            'location': location,
            'aqi': aqi,
            'pm25': pm25,
            'suitability_score': suitability_score,
            'suitability_level': suitability_level,
            'air_quality_data': air_quality,
            'location_data': air_data['location']
        }
    
    def generate_urban_planning_recommendations(self, suitability_data, forecast_data, forecast_days):
        """Generate AI-powered urban planning recommendations"""
        combined_data = {
            'current_suitability': suitability_data,
            'forecast_data': forecast_data.to_dict('records') if forecast_data is not None else []
        }
        context = f"""
        Real Estate and Urban Planning Analysis in India over the next {forecast_days} days:
        Provide recommendations based on current and forecasted air quality data for:
        - Pollution-resilient building designs
        - Optimal site selection for real estate
        - Green infrastructure integration
        - Zoning and land-use policies
        - Smart filter deployment in pollution hotspots
        - Community resilience strategies
        Focus on practical solutions for Indian urban environments to mitigate air pollution impacts.
        """
        return self.analyzer.analyze_with_ai(combined_data, context)
    
    def calculate_aqi(self, pm25):
        """Calculate AQI from PM2.5"""
        if pm25 <= 12: return int((50 / 12) * pm25)
        elif pm25 <= 35.4: return int(50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1))
        elif pm25 <= 55.4: return int(100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5))
        elif pm25 <= 150.4: return int(150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5))
        else: return min(300, int(200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)))

def main():
    st.set_page_config(
        page_title="AI-GIS Pollution Management Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-low { border-left: 5px solid #28a745; }
    .risk-moderate { border-left: 5px solid #ffc107; }
    .risk-high { border-left: 5px solid #fd7e14; }
    .risk-critical { border-left: 5px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 AI-GIS Pollution Management Platform</h1>
        <p>Comprehensive solution for Agriculture, Smart Cities, Healthcare, Sustainable Travel, and Real Estate</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API keys and navigation
    st.sidebar.title("🔑 API Configuration")
    weather_api_key = st.sidebar.text_input("WeatherAPI Key", type="password", placeholder="Enter your WeatherAPI key")
    groq_api_key = st.sidebar.text_input("Grok API Key (optional)", type="password", placeholder="Enter your Grok API key or leave blank for default")
    
    # Initialize analyzer with user-provided keys
    analyzer = PollutionAnalyzer(weather_api_key, groq_api_key)
    
    st.sidebar.title("🎯 Select Module")
    module = st.sidebar.selectbox(
        "Choose Analysis Module",
        ["🌾 Agriculture AI-GIS", "🏙️ Smart Cities Dashboard", "🏥 Healthcare Risk Assessment",
         "🗺️ Sustainable Travel & Eco-Tourism", "🏡 Real Estate & Urban Planning", "📊 Integrated Dashboard"]
    )
    
    if module == "🌾 Agriculture AI-GIS":
        agriculture_module(analyzer)
    elif module == "🏙️ Smart Cities Dashboard":
        smart_cities_module(analyzer)
    elif module == "🏥 Healthcare Risk Assessment":
        healthcare_module(analyzer)
    elif module == "🗺️ Sustainable Travel & Eco-Tourism":
        travel_eco_tourism_module(analyzer)
    elif module == "🏡 Real Estate & Urban Planning":
        real_estate_urban_planning_module(analyzer)
    else:
        integrated_dashboard(analyzer)

def agriculture_module(analyzer):
    st.header("🌾 Agriculture AI-GIS: Pollution-Resilient Farming")
    agriculture = AgricultureModule(analyzer)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        location = st.text_input("📍 Farm Location", value="Punjab, India")
    with col2:
        crop_type = st.selectbox("🌱 Crop Type", ['Wheat', 'Rice', 'Corn', 'Soybean', 'Cotton'])
    with col3:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    if st.button("🔍 Analyze Crop Impact", type="primary"):
        with st.spinner("Analyzing pollution impact on crops..."):
            impact_data = agriculture.predict_crop_impact(location, crop_type)
            forecast_data = analyzer.get_weather_forecast(location, days=14)
            forecast_df = analyzer.process_forecast_data(forecast_data, days=14) if forecast_data else None
            
            if impact_data:
                # Triggered Values and Sources
                st.subheader("🎯 Triggered Values and Sources")
                triggered_values = [
                    {
                        'Metric': f'{crop_type} Yield Loss',
                        'Value': f"{impact_data['total_yield_loss']:.1f}%",
                        'Threshold': 'Low (<2%), Moderate (2-5%), High (5-10%), Critical (>10%)',
                        'Source': 'WeatherAPI (2025); Mills et al. (2018)'
                    }
                ]
                df_triggered = pd.DataFrame(triggered_values)
                st.dataframe(df_triggered, use_container_width=True)
                st.markdown("""
                **Sources**:
                - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com).
                - xAI (2025): [https://api.groq.com](https://api.groq.com).
                - Mills et al. (2018): *Atmospheric Environment*, 191, 113-125. DOI: 10.1016/j.atmosenv.2018.07.031.
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    yield_loss = impact_data['total_yield_loss']
                    risk_class = "risk-low" if yield_loss < 2 else "risk-moderate" if yield_loss < 5 else "risk-high" if yield_loss < 10 else "risk-critical"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>Predicted Yield Loss</h3>
                        <h2>{yield_loss:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    location_data = impact_data['location_data']
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📍 Location</h4>
                        <p>{location_data['name']}, {location_data['region']}</p>
                        <p>Lat: {location_data['lat']:.2f}, Lon: {location_data['lon']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⏰ Analysis Time</h4>
                        <p>{current_time}</p>
                        <p>Real-time Data</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("🔬 Current Pollutant Impact Analysis")
                pollutant_data = [
                    {'Pollutant': pollutant.upper(), 'Concentration': f"{data['value']:.1f} μg/m³",
                     'Impact (%)': f"{data['impact_percent']:.1f}%", 'Risk Level': data['risk_level']}
                    for pollutant, data in impact_data['pollutant_impacts'].items()
                ]
                df_pollutants = pd.DataFrame(pollutant_data)
                st.dataframe(df_pollutants, use_container_width=True)
                
                fig = px.bar(df_pollutants, x='Pollutant', y='Impact (%)', color='Risk Level',
                             title=f"Pollution Impact on {crop_type} Yield",
                             color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107',
                                               'High': '#fd7e14', 'Critical': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
                
                if forecast_df is not None:
                    st.subheader("📈 14-Day Pollutant Forecast")
                    st.dataframe(forecast_df[['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                    
                    fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                    for i, pollutant in enumerate(pollutants):
                        row, col = positions[i]
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[pollutant], mode='lines+markers', name=pollutant),
                                              row=row, col=col)
                    fig_forecast.update_layout(height=800, title_text="14-Day Pollutant Forecast", showlegend=False)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader(f"🤖 AI-Powered Farming Recommendations (Next {forecast_days} Days)")
                with st.spinner("Generating recommendations..."):
                    recommendations = agriculture.generate_farming_recommendations(
                        impact_data, forecast_df[:forecast_days] if forecast_df is not None else None, crop_type, forecast_days)
                    st.markdown(recommendations)

def smart_cities_module(analyzer):
    st.header("🏙️ Smart Cities: Predictive AI for Air Quality Management")
    smart_cities = SmartCitiesModule(analyzer)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_cities = st.multiselect("🏙️ Select Cities for Monitoring",
                                        list(smart_cities.indian_cities.keys()),
                                        default=['Delhi', 'Mumbai', 'Bangalore', 'Chennai'])
    with col2:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    if st.button("🌍 Generate Smart City Dashboard", type="primary"):
        with st.spinner("Fetching real-time air quality data..."):
            cities_data = {city: analyzer.get_air_quality_data(city) for city in selected_cities}
            cities_forecast = {city: analyzer.get_weather_forecast(city, days=14) for city in selected_cities}
            cities_forecast_df = {city: analyzer.process_forecast_data(data, days=14) if data else None for city, data in cities_forecast.items()}
            
            # Triggered Values and Sources
            comparison_data = [
                {'City': city, 'PM2.5': data['current']['air_quality'].get('pm2_5', 0),
                 'PM10': data['current']['air_quality'].get('pm10', 0),
                 'O3': data['current']['air_quality'].get('o3', 0),
                 'NO2': data['current']['air_quality'].get('no2', 0),
                 'SO2': data['current']['air_quality'].get('so2', 0),
                 'CO': data['current']['air_quality'].get('co', 0),
                 'AQI': smart_cities.calculate_aqi(data['current']['air_quality'].get('pm2_5', 0))}
                for city, data in cities_data.items() if data and 'current' in data
            ]
            st.subheader("🎯 Triggered Values and Sources")
            triggered_values = [
                {
                    'Metric': f"AQI ({data['City']})",
                    'Value': f"{data['AQI']:.1f}",
                    'Threshold': 'Good (<50), Moderate (50-100), Poor (100-200), Hazardous (>200)',
                    'Source': 'WeatherAPI (2025); U.S. EPA (2023)'
                } for data in comparison_data
            ]
            df_triggered = pd.DataFrame(triggered_values)
            st.dataframe(df_triggered, use_container_width=True)
            st.markdown("""
            **Sources**:
            - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com).
            - xAI (2025): [https://api.groq.com](https://api.groq.com).
            - U.S. EPA (2023): Air Quality Index (AQI) Basics. [https://www.airnow.gov](https://www.airnow.gov).
            """)
            
            st.subheader("🗺️ Real-Time Air Quality Map")
            pollution_map = smart_cities.create_pollution_map(cities_data)
            folium_static(pollution_map, width=1200, height=600)
            
            st.subheader("📊 Current Air Quality Dashboard")
            if comparison_data:
                df_cities = pd.DataFrame(comparison_data)
                cols = st.columns(len(selected_cities))
                for i, city in enumerate(selected_cities):
                    if i < len(cols) and not df_cities[df_cities['City'] == city].empty:
                        city_data = df_cities[df_cities['City'] == city].iloc[0]
                        with cols[i]:
                            aqi = city_data['AQI']
                            pm25 = city_data['PM2.5']
                            risk_class = "risk-low" if aqi < 50 else "risk-moderate" if aqi < 100 else "risk-high" if aqi < 200 else "risk-critical"
                            st.markdown(f"""
                            <div class="metric-card {risk_class}">
                                <h5>{city}</h5>
                                <h3>AQI: {aqi}</h3>
                                <p>PM2.5: {pm25:.1f} μg/m³</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_pm25 = px.bar(df_cities, x='City', y='PM2.5', title="PM2.5 Levels Across Cities",
                                     color='PM2.5', color_continuous_scale='Reds')
                    st.plotly_chart(fig_pm25, use_container_width=True)
                with col2:
                    fig_aqi = px.bar(df_cities, x='City', y='AQI', title="Air Quality Index Comparison",
                                    color='AQI', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig_aqi, use_container_width=True)
                
                st.subheader("🔬 Current Pollutant Analysis")
                fig = make_subplots(rows=2, cols=3, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'),
                                   specs=[[{"secondary_y": False}] * 3] * 2)
                pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
                positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
                for i, pollutant in enumerate(pollutants):
                    row, col = positions[i]
                    fig.add_trace(go.Bar(x=df_cities['City'], y=df_cities[pollutant], name=pollutant, showlegend=False),
                                 row=row, col=col)
                fig.update_layout(height=600, title_text="Comprehensive Pollutant Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📈 14-Day Pollutant Forecast")
                for city in selected_cities:
                    if cities_forecast_df.get(city) is not None:
                        st.write(f"**{city}**")
                        st.dataframe(cities_forecast_df[city][['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                        fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                        forecast_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                        forecast_positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                        for i, pollutant in enumerate(forecast_pollutants):
                            row, col = forecast_positions[i]
                            fig_forecast.add_trace(go.Scatter(x=cities_forecast_df[city]['date'], y=cities_forecast_df[city][pollutant],
                                                             mode='lines+markers', name=pollutant), row=row, col=col)
                        fig_forecast.update_layout(height=800, title_text=f"14-Day Pollutant Forecast for {city}", showlegend=False)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader(f"🤖 AI-Powered City Management Recommendations (Next {forecast_days} Days)")
                with st.spinner("Generating recommendations..."):
                    recommendations = smart_cities.generate_city_recommendations(
                        comparison_data, cities_forecast_df.get(selected_cities[0]), forecast_days)
                    st.markdown(recommendations)

def healthcare_module(analyzer):
    st.header("🏥 Healthcare: Personalized Air Quality Risk Management")
    healthcare = HealthcareModule(analyzer)
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        location = st.text_input("📍 Your Location", value="Delhi, India")
    with col2:
        age_group = st.selectbox("👤 Age Group", ['child', 'adult', 'elderly'])
    with col3:
        conditions = st.multiselect("🏥 Health Conditions", ['asthma', 'heart_disease', 'diabetes'])
    with col4:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    if st.button("🩺 Assess Health Risk", type="primary"):
        with st.spinner("Analyzing health risks..."):
            risk_assessment = healthcare.assess_health_risk(location, age_group, conditions)
            forecast_data = analyzer.get_weather_forecast(location, 14)
            forecast_df = analyzer.process_forecast_data(forecast_data, days=14) if forecast_data else None
            
            if risk_assessment:
                # Triggered Values and Sources
                st.subheader("🎯 Triggered Values and Sources")
                triggered_values = [
                    {
                        'Metric': 'Overall Risk Score',
                        'Value': f"{risk_assessment['overall_risk_score']:.1f}/10",
                        'Threshold': 'Low (<2), Moderate (2-4), High (4-6), Very High (6-8), Hazardous (>8)',
                        'Source': 'WeatherAPI (2025); WHO (2021)'
                    }
                ]
                df_triggered = pd.DataFrame(triggered_values)
                st.dataframe(df_triggered, use_container_width=True)
                st.markdown("""
                **Sources**:
                - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com).
                - xAI (2025): [https://api.groq.com](https://api.groq.com).
                - WHO (2021): WHO Global Air Quality Guidelines. [https://www.who.int](https://www.who.int).
                """)
                
                overall_risk = risk_assessment['overall_risk_score']
                risk_level = risk_assessment['overall_risk_level']
                risk_colors = {'Low': '#28a745', 'Moderate': '#ffc107', 'High': '#fd7e14',
                              'Very High': '#dc3545', 'Hazardous': '#6f42c1'}
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="background: {risk_colors.get(risk_level, '#17a2b8')}; 
                                padding: 2rem; border-radius: 10px; color: white; text-align: center;">
                        <h2>Health Risk Level: {risk_level}</h2>
                        <h3>Risk Score: {overall_risk:.1f}/10</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>👤 Profile</h4>
                        <p>Age Group: {age_group.title()}</p>
                        <p>Conditions: {', '.join(conditions) if conditions else 'None'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📍 Location</h4>
                        <p>{location}</p>
                        <p>Real-time Analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("🔬 Current Pollutant-Specific Health Risks")
                risk_data = [
                    {'Pollutant': pollutant.upper(), 'Concentration': f"{data['value']:.1f} μg/m³",
                     'Risk Score': f"{data['risk_score']:.1f}/10", 'Risk Level': data['risk_level']}
                    for pollutant, data in risk_assessment['pollutant_risks'].items()
                ]
                df_risks = pd.DataFrame(risk_data)
                st.dataframe(df_risks, use_container_width=True)
                
                fig_risk = px.bar(df_risks, x='Pollutant', y='Risk Score', color='Risk Level',
                                 title="Health Risk by Pollutant",
                                 color_discrete_map={'Low': '#28a745', 'Moderate': '#ffc107',
                                                   'High': '#fd7e14', 'Very High': '#dc3545',
                                                   'Hazardous': '#6f42c1'})
                st.plotly_chart(fig_risk, use_container_width=True)
                
                if forecast_df is not None:
                    st.subheader("📈 14-Day Pollutant Forecast")
                    st.dataframe(forecast_df[['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                    
                    fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                    for i, pollutant in enumerate(pollutants):
                        row, col = positions[i]
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[pollutant], mode='lines+markers', name=pollutant),
                                              row=row, col=col)
                    fig_forecast.update_layout(height=800, title_text="14-Day Pollutant Forecast", showlegend=False)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("💡 Basic Personalized Health Recommendations")
                for i, rec in enumerate(risk_assessment['recommendations'], 1):
                    st.markdown(f"**{i}.** {rec}")
                
                st.subheader(f"🤖 AI Health Insights (Next {forecast_days} Days)")
                with st.spinner("Generating health insights..."):
                    ai_insights = healthcare.generate_ai_health_recommendations(
                        risk_assessment, forecast_df[:forecast_days] if forecast_df is not None else None, age_group, conditions, forecast_days)
                    st.markdown(ai_insights)

def travel_eco_tourism_module(analyzer):
    st.header("🗺️ Sustainable Travel & Eco-Tourism: Low-Pollution Travel Planning")
    travel = TravelEcoTourismModule(analyzer)
    
    st.markdown("""
    **Why It Matters**: Air pollution can reduce tourist arrivals by 10-15% in heavily polluted urban areas (UNWTO, 2019). This module helps travelers plan low-pollution routes and discover clean-air destinations.
    """)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_city = st.selectbox("📍 Start City", list(travel.indian_cities.keys()), index=0)
    with col2:
        end_city = st.selectbox("📍 End City", list(travel.indian_cities.keys()), index=1)
    with col3:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    selected_cities = st.multiselect("🏙️ Select Cities for Eco-Tourism Analysis",
                                    list(travel.indian_cities.keys()),
                                    default=['Bangalore', 'Pune', 'Jaipur'])
    
    if st.button("🗺️ Analyze Travel & Eco-Tourism", type="primary"):
        with st.spinner("Analyzing travel routes and eco-tourism destinations..."):
            # Route optimization
            route_data = travel.optimize_low_pollution_route(start_city, end_city)
            start_forecast = analyzer.get_weather_forecast(start_city, 14)
            end_forecast = analyzer.get_weather_forecast(end_city, 14)
            start_forecast_df = analyzer.process_forecast_data(start_forecast, 14) if start_forecast else None
            end_forecast_df = analyzer.process_forecast_data(end_forecast, 14) if end_forecast else None
            
            # Triggered Values and Sources
            st.subheader("🎯 Triggered Values and Sources")
            triggered_values = [
                {
                    'Metric': 'Route Pollution Score',
                    'Value': f"{route_data['route_score']:.1f}" if route_data else 'N/A',
                    'Threshold': 'Low (<50), Moderate (50-100), High (>100)',
                    'Source': 'WeatherAPI (2025); UNWTO (2019)'
                }
            ]
            df_triggered = pd.DataFrame(triggered_values)
            st.dataframe(df_triggered, use_container_width=True)
            st.markdown("""
            **Sources**:
            - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com).
            - xAI (2025): [https://api.groq.com](https://api.groq.com).
            - UNWTO (2019): Tourism and the Sustainable Development Goals – Journey to 2030. [https://www.unwto.org](https://www.unwto.org).
            """)
            
            st.subheader("🚗 Low-Pollution Route Analysis")
            if route_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    risk_class = "risk-low" if route_data['route_score'] < 50 else "risk-moderate" if route_data['route_score'] < 100 else "risk-high"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>Route Pollution Score</h3>
                        <h2>{route_data['route_score']:.1f}</h2>
                        <p>Status: {route_data['route_status']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📍 Start: {start_city}</h4>
                        <p>AQI: {route_data['start_aqi']:.1f}</p>
                        <p>Lat: {route_data['start_location']['lat']:.2f}, Lon: {route_data['start_location']['lon']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📍 End: {end_city}</h4>
                        <p>AQI: {route_data['end_aqi']:.1f}</p>
                        <p>Lat: {route_data['end_location']['lat']:.2f}, Lon: {route_data['end_location']['lon']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Eco-tourism destinations
            st.subheader("🌳 Clean-Air Destinations")
            clean_destinations = travel.identify_clean_air_destinations(selected_cities)
            if clean_destinations:
                clean_data = [
                    {'City': d['city'], 'AQI': d['aqi'], 'Latitude': d['location']['lat'], 'Longitude': d['location']['lon']}
                    for d in clean_destinations
                ]
                df_clean = pd.DataFrame(clean_data)
                st.dataframe(df_clean, use_container_width=True)
                
                fig = px.scatter(df_clean, x='Longitude', y='Latitude', color='AQI', size='AQI',
                                title="Clean-Air Destinations", color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            
            # Pollution hotspots
            st.subheader("🔥 Pollution Hotspots Map")
            cities_data = {city: analyzer.get_air_quality_data(city) for city in selected_cities}
            hotspot_map = travel.map_pollution_hotspots(cities_data)
            folium_static(hotspot_map, width=1200, height=600)
            
            # Forecast for route cities
            st.subheader("📈 14-Day Pollutant Forecast for Route")
            for city, forecast_df in [(start_city, start_forecast_df), (end_city, end_forecast_df)]:
                if forecast_df is not None:
                    st.write(f"**{city}**")
                    st.dataframe(forecast_df[['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                    fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                    for i, pollutant in enumerate(['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']):
                        row, col = positions[i]
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[pollutant], mode='lines+markers', name=pollutant),
                                              row=row, col=col)
                    fig_forecast.update_layout(height=800, title_text=f"14-Day Forecast for {city}", showlegend=False)
                    st.plotly_chart(fig_forecast, use_container_width=True)
            
            # AI recommendations
            st.subheader(f"🤖 AI-Powered Travel Recommendations (Next {forecast_days} Days)")
            with st.spinner("Generating travel recommendations..."):
                recommendations = travel.generate_travel_recommendations(
                    route_data, start_forecast_df, end_forecast_df, forecast_days)
                st.markdown(recommendations)

def real_estate_urban_planning_module(analyzer):
    st.header("🏡 Real Estate & Urban Planning: Pollution-Resilient Infrastructure")
    real_estate = RealEstateUrbanPlanningModule(analyzer)
    
    st.markdown("""
    **Why It Matters**: GIS-based suitability analysis optimizes real estate development by identifying low-pollution sites and deploying smart filters in hotspots.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        location = st.text_input("📍 Development Location", value="Mumbai, India")
    with col2:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    if st.button("🏗️ Analyze Site Suitability", type="primary"):
        with st.spinner("Analyzing site suitability..."):
            suitability_data = real_estate.assess_site_suitability(location)
            forecast_data = analyzer.get_weather_forecast(location, 14)
            forecast_df = analyzer.process_forecast_data(forecast_data, 14) if forecast_data else None
            
            if suitability_data:
                # Triggered Values and Sources
                st.subheader("🎯 Triggered Values and Sources")
                triggered_values = [
                    {
                        'Metric': 'Suitability Score',
                        'Value': f"{suitability_data['suitability_score']:.1f}/100",
                        'Threshold': 'High (>80), Moderate (50-80), Low (<50)',
                        'Source': 'WeatherAPI (2025); U.S. EPA (2023)'
                    }
                ]
                df_triggered = pd.DataFrame(triggered_values)
                st.dataframe(df_triggered, use_container_width=True)
                st.markdown("""
                **Sources**:
                - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com).
                - xAI (2025): [https://api.groq.com](https://api.groq.com).
                - U.S. EPA (2023): AI-GroQAir Quality Index (AQI) Basics. [https://www.airnow.gov](https://www.airnow.gov).
                """)
                
                col1, col2, col3 = st.columns(3)
                with col2:
                    suitability_score = suitability_data['suitability_score']
                    risk_class = "risk-low" if suitability_score > 80 else "risk-moderate" if suitability_score > 50 else "risk-high"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>Site Suitability Score</h3>
                        <h2>{suitability_score:.1f}/100</h2>
                        <p>Level: {suitability_data['suitability_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📍 Location</h4>
                        <p>{suitability_data['location_data']['name']}, {suitability_data['location_data']['region']}</p>
                        <p>AQI: {suitability_data['aqi']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⏰ Analysis Time</h4>
                        <p>{current_time}</p>
                        <p>Real-time Data</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("🔬 Current Air Quality Analysis")
                air_data = [
                    {'Pollutant': 'PM2.5', 'Concentration': f"{suitability_data['pm25']:.1f} μg/m³", 'AQI': suitability_data['aqi']}
                ]
                df_air = pd.DataFrame(air_data)
                st.dataframe(df_air, use_container_width=True)
                
                fig = px.bar(df_air, x='Pollutant', y='AQI', title="Air Quality Impact on Site", 
                             color='AQI', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
                if forecast_df is not None:
                    st.subheader("📈 14-Day Pollutant Forecast")
                    st.dataframe(forecast_df[['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                    
                    fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                    for i in range(1, len(pollutants)):
                        row, col = positions[i]
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[pollutants[i]], mode='lines+markers', name=pollutants[i]),
                                              row=row, col=col)
                    fig_forecast.update_layout(height=800, title_text="14-Day Pollutant Forecast", showlegend=False)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader(f"🤖 AI-Powered urban planning recommendations (Next {forecast_days} Days)")
                with st.spinner("Generating recommendations..."):
                    recommendations = real_estate.generate_urban_planning_recommendations(
                        suitability_data, forecast_df[:forecast_days] if forecast_data is not None else None, forecast_days)
                    st.markdown(recommendations)

def integrated_dashboard(analyzer):
    st.header("📊 Integrated Multi-Domain Dashboard")
    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input("📍 Analysis Location", value="Delhi, India")
    with col2:
        forecast_days = st.selectbox("📅 Forecast Period", [3, 7, 14], index=2)
    
    if st.button("🚖 Generate Comprehensive Analysis", type="primary"):
        with st.spinner("Generating comprehensive analysis..."):
            agriculture = AgricultureModule(analyzer)
            smart_cities = SmartCitiesModule(analyzer)
            healthcare = HealthcareModule(analyzer)
            travel = TravelEcoTourismModule(analyzer)
            real_estate = RealEstateUrbanPlanningModule(analyzer)

        air_data = analyzer.get_air_quality_data(location)
        forecast_data = analyzer.get_weather_forecast(location, 14)
        forecast_df = analyzer.process_forecast_data(forecast_data, 14) if forecast_data else None

        if air_data:
            air_quality = air_data['current'].get('air_quality', {})
            location_info = air_data['location']
            pm25 = air_quality.get('pm2_5', 0)
            pm10 = air_quality.get('pm10', 0)
            o3 = air_quality.get('o3', 0)
            no2 = air_quality.get('no2', 0)
            so2 = air_quality.get('so2', 0)
            aqi = smart_cities.calculate_aqi(pm25)
            
            # Triggered Values and Sources
            st.subheader("🎯 Triggered Values Across All Modules")
            triggered_values = [
                {
                    'Module': 'Agriculture',
                    'Metric': 'Wheat Yield Loss',
                    'Value': f"{agriculture.predict_crop_impact(location, 'wheat')['total_yield_loss']:.1f}%" if agriculture.predict_crop_impact(location, 'wheat') else 'N/A',
                    'Threshold': 'Low (<3%), Moderate (3-7%), High (>7%)',
                    'Source': 'WeatherAPI (2025); Mills et al. (2018)'
                },
                {
                    'Module': 'Agriculture',
                    'Metric': 'Rice Yield Loss',
                    'Value': f"{agriculture.predict_crop_impact(location, 'rice')['total_yield_loss']:.1f}%" if agriculture.predict_crop_impact(location, 'rice') else 'N/A',
                    'Threshold': 'Low (<3%), Moderate (3-7%), High (>7%)',
                    'Source': 'WeatherAPI (2025); Mills et al. (2018)'
                },
                {
                    'Module': 'Smart Cities',
                    'Metric': 'AQI',
                    'Value': f"{aqi:.1f}",
                    'Threshold': 'Good (<50), Moderate (50-100), Poor (100-200), Hazardous (>200)',
                    'Source': 'WeatherAPI (2025); U.S. EPA (2023)'
                },
                {
                    'Module': 'Healthcare',
                    'Metric': 'Adult Risk Score',
                    'Value': f"{healthcare.assess_health_risk(location, 'adult')['overall_risk_score']:.1f}/10" if healthcare.assess_health_risk(location, 'adult') else 'N/A',
                    'Threshold': 'Low (<3), Moderate (3-6), High (>6)',
                    'Source': 'WeatherAPI (2025)'
                },
                {
                    'Module': 'Healthcare',
                    'Metric': 'Elderly Risk Score',
                    'Value': f"{healthcare.assess_health_risk(location, 'elderly')['overall_risk_score']:.1f}/10" if healthcare.assess_health_risk(location, 'elderly') else 'N/A',
                    'Threshold': 'Low (<3), Moderate (3-6), High (>6)',
                    'Source': 'WeatherAPI (2025); WHO (2021)'
                },
                {
                    'Module': 'Travel & Eco-Tourism',
                    'Metric': 'Route Pollution Score',
                    'Value': f"{travel.optimize_low_pollution_route(location, 'Pune')['route_score']:.1f}" if travel.optimize_low_pollution_route(location, 'Pune') else 'N/A',
                    'Threshold': 'Low (<50), Moderate (50-100), High (>100)',
                    'Source': 'WeatherAPI (2025); UNWTO (2019)'
                },
                {
                    'Module': 'Real Estate',
                    'Metric': 'Suitability Score',
                    'Value': f"{real_estate.assess_site_suitability(location)['suitability_score']:.1f}/100" if real_estate.assess_site_suitability(location) else 'N/A',
                    'Threshold': 'High (>80), Moderate (50-80), Low (<50)',
                    'Source': 'WeatherAPI (2025); U.S. EPA (2023)'
                }
            ]
            df_triggered = pd.DataFrame(triggered_values)
            st.dataframe(df_triggered, use_container_width=True)
            st.markdown("""
            **Sources**:
            - WeatherAPI (2025): [http://api.weatherapi.com](http://api.weatherapi.com). Accessed May 25, 2025, 02:28 AM IST.
            - xAI (2025): [https://api.groq.com](https://api.groq.com). Accessed May 26, 2025, 02:28 AM IST.
            - U.S. EPA (2023): Air Quality Index (AQI) Basics. [https://www.airnow.gov](https://www.airnow.gov).
            - WHO (2021): WHO Global Air Quality Guidelines. [https://www.who.int](https://www.who.int).
            - UNWTO (2019): Tourism and the Sustainable Development Goals – Journey to 2030. [https://www.unwto.org](https://www.unwto.org).
            - Mills et al., (2018): Ozone pollution: Impacts on crop yields. *Atmospheric Environment*, 191, 113-128. DOI:10.1016/j.atmosenv.2018.07.031.
            """)
            
            st.subheader("🌍 Current Location Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("PM2.5", f"{pm25:.1f} μg/m³")
            with col2: st.metric("PM10", f"{pm10:.1f} μg/m³")
            with col3: st.metric("O3", f"{o3:.1f} μg/m³")
            with col4: st.metric("NO2", f"{no2:.1f} μg/m³")
            with col5: st.metric("SO2", f"{so2:.1f} μg/m³")

            # Show AQI as a metric
            st.metric("AQI", f"{aqi:.1f}")

            # Show forecast if available
            if forecast_df is not None:
                st.subheader("📈 14-Day Pollutant Forecast")
                st.dataframe(forecast_df[['date', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']], use_container_width=True)
                fig_forecast = make_subplots(rows=3, cols=2, subplot_titles=('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO'))
                pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
                for i, pollutant in enumerate(pollutants):
                    row, col = positions[i]
                    fig_forecast.add_trace(
                        go.Scatter(x=forecast_df['date'], y=forecast_df[pollutant], mode='lines+markers', name=pollutant),
                        row=row, col=col
                    )
                fig_forecast.update_layout(height=800, title_text="14-Day Pollutant Forecast", showlegend=False)
                st.plotly_chart(fig_forecast, use_container_width=True)

            # AI-powered summary/recommendations
            st.subheader(f"🤖 AI-Powered Summary & Recommendations (Next {forecast_days} Days)")
            with st.spinner("Generating AI-powered summary..."):
                # Example: combine all relevant data for AI analysis
                combined_data = {
                    "air_quality": air_quality,
                    "location_info": location_info,
                    "forecast": forecast_df.to_dict('records') if forecast_df is not None else [],
                    "triggered_values": triggered_values
                }
                context = f"""
                Integrated multi-domain air quality and pollution impact analysis for {location} over the next {forecast_days} days.
                Provide a summary and actionable recommendations for agriculture, smart cities, healthcare, travel, and real estate.
                """
                ai_summary = analyzer.analyze_with_ai(combined_data, context)
                st.markdown(ai_summary)

if __name__ == "__main__":
    main()
    # Footer
    st.markdown("Powered by [WeatherAPI.com](https://www.weatherapi.com/) and [GroqAPI](https://groq.com/)| Built with Streamlit")
    st.markdown("Prepared by: Dr. Anil Kumar Singh | [LinkedIn](https://www.linkedin.com/in/anil-kumar-singh-phd-b192554a/)")

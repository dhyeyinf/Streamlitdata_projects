import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.write("""
# California House Price Prediction App
""")
st.write('---')

# Load the dataset
file_path = 'housing.csv'
housing_data = pd.read_csv(file_path)

# Encode 'ocean_proximity' since it's categorical
label_encoder = LabelEncoder()
housing_data['ocean_proximity'] = label_encoder.fit_transform(housing_data['ocean_proximity'])

# Separate features (X) and target (Y)
X = housing_data.drop('median_house_value', axis=1)
Y = housing_data['median_house_value']

# Sidebar
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider('Longitude', X.longitude.min(), X.longitude.max(), X.longitude.mean())
    latitude = st.sidebar.slider('Latitude', X.latitude.min(), X.latitude.max(), X.latitude.mean())
    housing_median_age = st.sidebar.slider('Housing Median Age', X.housing_median_age.min(), X.housing_median_age.max(), X.housing_median_age.mean())
    total_rooms = st.sidebar.slider('Total Rooms', X.total_rooms.min(), X.total_rooms.max(), X.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('Total Bedrooms', X.total_bedrooms.min(), X.total_bedrooms.max(), X.total_bedrooms.mean())
    population = st.sidebar.slider('Population', X.population.min(), X.population.max(), X.population.mean())
    households = st.sidebar.slider('Households', X.households.min(), X.households.max(), X.households.mean())
    median_income = st.sidebar.slider('Median Income', X.median_income.min(), X.median_income.max(), X.median_income.mean())
    ocean_proximity = st.sidebar.selectbox('Ocean Proximity', label_encoder.classes_)
    ocean_proximity_encoded = label_encoder.transform([ocean_proximity])[0]
    
    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity_encoded}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Median House Value')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

import streamlit as st
import pickle
import pandas as pd

# List of IPL Teams
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Gujarat Titans',
 'Lucknow Super Giants']

# List of Cities that matches are played
cities = ['Kolkata', 'Durban', 'Delhi', 'Jaipur', 'Lucknow', 'Mohali',
       'Chennai', 'Pune', 'Hyderabad', 'Kochi', 'Cape Town', 'Mumbai',
       'East London', 'Bangalore', 'Johannesburg', 'Centurion', 'Raipur',
       'Ranchi', 'Kanpur', 'Visakhapatnam', 'Cuttack', 'Indore',
       'Dharamsala', 'Nagpur', 'Rajkot', 'Port Elizabeth', 'Ahmedabad',
       'Abu Dhabi', 'Sharjah', 'Bloemfontein', 'Kimberley']

# Load the pre-trained machine learning model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set the title of the Streamlit app
st.title('IPL Win Predictor')

# Create two columns for team selection
col1, col2 = st.columns(2)

# Team selection dropdown for batting team
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))  # Alphabetically teams are sorted

# Team selection dropdown for bowling team
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

# Dropdown for selecting the host city
selected_city = st.selectbox('Select Host City', sorted(cities))

# Number input for target score
target = st.number_input('Target')

# Create three columns for additional input parameters
col3, col4, col5 = st.columns(3)

# Number input for current score
with col3:
    score = st.number_input('Current Score')

# Number input for overs completed
with col4:
    overs = st.number_input('Overs Completed')

# Number input for wickets fallen
with col5:
    wickets_gone = st.number_input('Wickets Gone')

# Button to trigger the prediction
if st.button('Predict Probability'):

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets_gone
    crr = score / overs
    rrr = (runs_left*6) / balls_left
    
    # Create a DataFrame with the calculated parameters
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 
                  'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 
                  'wickets_left': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr] }) # Parsing a dictionary
    
    #st.table(input_df)  # Shows the table of the data of input_df
     
    # Predict win and loss probabilities using the pre-trained model
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    # Display the predicted probabilities
    st.header(batting_team + " - " + str(round(win * 100)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100)) + "%")

    #st.text(result)   # Display the result
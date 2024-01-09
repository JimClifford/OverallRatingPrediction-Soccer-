#DEPLOYMENT PART 2

import streamlit as st
import pandas as pd
import pickle

# Load your trained model
filename1 = 'model1.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))

# Function to predict player rating
def predict_player_rating(player_data):
    player_rating = loaded_model1.predict(player_data)
    return player_rating

# Create a Streamlit web app
st.title("Player Rating Predictor")

# Create input fields for player data
st.write("Enter Player Data: ")
player_data = {}  # Create a dictionary to store user input 

# Define the feature names in Xtest and their corresponding custom labels
features = {
    'lf': 'Left Forward Rating', 'cf': 'Center Forward Rating', 'rf': 'Right Forward Rating',
    'lam': 'Left Attacking Midfield Rating', 'cam': 'Center Attacking Midfield Rating', 'ram': 'Right Attacking Midfield Rating',
    'lm': 'Left Midfield Rating', 'lcm': 'Left Central Midfield Rating', 'cm': 'Central Midfield Rating',
    'rcm': 'Right Central Midfield Rating', 'rm': 'Right Midfield Rating',
    'potential': 'Potential Rating', 'release_clause_eur': 'Release Clause (1 - 1 000 000 000 EUR)',
    'passing': 'Passing Rating', 'movement_reactions': 'Movement Reactions Rating',
    'mentality_composure': 'Mentality Composure Rating'
}

program_conf = 0.9611822822603266 # the score from the MidSemester Project code that shows the level of the model1's accuracy of predictions in relation to the actual values


for feature in features:
    if (feature == 'release_clause_eur'):
        player_data[feature] = st.number_input(f"{features[feature]}", min_value=0.0, max_value=1000000000.0)
    else:
        player_data[feature] = st.number_input(f"{features[feature]}", min_value=0.0, max_value=100.0)
 

if st.button("Predict"):
    # Convert the input data to a DataFrame for prediction
    input_data = pd.DataFrame(player_data, index=[0])

    # Call the prediction function
    player_rating = predict_player_rating(input_data)

    # Display the predicted player rating
    st.write(f"Predicted Player Rating: {player_rating[0]}")
    st.write("Program Confidence: " + str(program_conf))

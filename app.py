import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data
def generate_data(n):
    data = {
        'Company': [f'Company{str(i)}' for i in range(1, n + 1)],
        'Service Now': np.random.choice(['Remote', 'Localized', 'Hybrid'], n),
        'Raise Funds': np.random.choice([True, False], n),
        'Company Revenue ($M)': np.random.uniform(1, 100, n),
        'Number of Employees': np.random.randint(1, 1000, n),
        'Industry': np.random.choice(
            ['Tech', 'Finance', 'Health', 'Retail', 'Education'], n
        ),
        'Location': np.random.choice(
            ['USA', 'UK', 'Canada', 'Germany', 'France'], n
        ),
        'Multiple Positions Open': np.random.choice([True, False], n),
        'Job Title': np.random.choice(
            ['Manager', 'Engineer', 'Analyst', 'Sales', 'Developer'], n
        ),
        'Job Location': np.random.choice(
            ['USA', 'UK', 'Canada', 'Germany', 'France'], n
        ),
        'Website Access': np.random.randint(1, 1000, n),
        'Downloads': np.random.choice([True, False], n),
        'Emails (Close)': np.random.choice([True, False], n),
        'Frequency of Interactions': np.random.randint(1, 100, n),
        'Response Times (hours)': np.random.uniform(1, 48, n),
        'Type of Questions': np.random.choice(
            ['General', 'Technical', 'Financial', 'Operational', 'Other'], n
        ),
        'LTV': np.random.uniform(6, 24, n),  # months
        'VAR': np.random.uniform(0.3, 0.45, n),  # profitability
        'Conversions': np.random.uniform(0, 1, n),  # success rate
    }

    return pd.DataFrame(data)

# Calculate the LoC score per lead
def calculate_loc_score(df):
    # Select the explanatory variables
    X = df[['Service Now', 'Raise Funds', 'Company Revenue ($M)', 'Number of Employees', 'Industry',
            'Location', 'Multiple Positions Open', 'Job Title', 'Job Location', 'Website Access',
            'Downloads', 'Emails (Close)', 'Frequency of Interactions', 'Response Times (hours)', 
            'Type of Questions']]
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X)
    
    # Select the dependent variables
    y = df[['LTV', 'VAR', 'Conversions']]
    
    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Train the ANN
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    model.fit(X_scaled, y_scaled)
    
    # Use the model to predict the scaled LTV, VAR, and Conversions
    predictions_scaled = model.predict(X_scaled)
    
    # Reverse the scaling of the predictions
    predictions = scaler_y.inverse_transform(predictions_scaled)
    df['Predicted LTV'] = predictions[:, 0]
    df['Predicted VAR'] = predictions[:, 1]
    df['Predicted Conversions'] = predictions[:, 2]
    
    # Calculate the LoC score as the mean of the predicted values
    df['LoC Score'] = np.mean(predictions, axis=1)
    
    # Scale the LoC Score to be between 0 and 1
    scaler = MinMaxScaler()
    df['LoC Score'] = scaler.fit_transform(df[['LoC Score']])
    
    return df

# Recommend the type of action to calibrate the lead
def recommend_action(loc_score):
    if loc_score < 0.3:
        return 'Investigative Actions'
    elif loc_score < 0.6:
        return 'Nurturing'
    else:
        return 'Opportunity'

# Adjust the weights based on the predicted values
def adjust_weights(df):
    df['Recommended Action'] = df['LoC Score'].apply(recommend_action)
    return df

# Create the heatmap
def create_heatmap(df):
    # Select the columns to be used in the heatmap
    heatmap_data = df[['Company', 'LoC Score', 'Predicted LTV', 'Predicted VAR', 'Predicted Conversions']]

    # Set the Company column as the index
    heatmap_data.set_index('Company', inplace=True)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
    plt.title('Heatmap of LoC Score and Predicted Values')
    plt.xlabel('Variables')
    plt.ylabel('Company')
    st.pyplot(fig)

# Main function to run the app
def main():
    st.title('LoC Algorithm Prototype')

    st.write('## Generate Random Data')
    num_companies = st.slider('Number of Companies', min_value=10, max_value=100, value=50)
    df = generate_data(num_companies)
    st.dataframe(df)

    st.write('## Calculate LoC Score')
    df = calculate_loc_score(df)
    st.dataframe(df)

    st.write('## Adjust Weights')
    df = adjust_weights(df)
    st.dataframe(df)

    st.write('## Explanation of Variables')
    st.write('LoC is an algorithm that calculates the lead calibration: what is the potential LTV, VAR, and Conversion rate and what actions to take to improve the LoC: investigate, nurturing, opportunity.')
    st.write('1. The LTV (Lifetime Value) is the month between 6 to 24 months.')
    st.write('2. VAR (Value At Risk) is the profitability between 30% to 45%.')
    st.write('3. Conversion rate is the success rate for a specific profile.')

# Run the app
if __name__ == '__main__':
    main()


# LoC Algorithm Prototype

## Overview
This repository contains a Streamlit app that demonstrates a prototype of the LoC (Level of Confidence) algorithm. The LoC algorithm is used in lead generation to predict the potential Lifetime Value (LTV), Value At Risk (VAR), and Conversion rate of leads and recommend actions to improve the Lead calibration score (LoC Score).

## How It Works
The app generates random data for a specified number of companies, then uses an Artificial Neural Network (ANN) to predict the potential LTV, VAR, and Conversion rate for each company. The LoC Score is then calculated as the mean of the predicted values and scaled to be between 0 and 1. Based on the LoC Score, the app recommends actions to improve the LoC Score: Investigative Actions, Nurturing, or Opportunity.

## Installation
1. Clone this repository.
2. Create a virtual environment and activate it.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the app using `streamlit run app.py`.

## Usage
1. Open the app in your browser.
2. Use the slider to select the number of companies to generate data for.
3. The app will display the generated data, the calculated LoC Score, the recommended actions, and a heatmap of the LoC Score and predicted values for each company.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

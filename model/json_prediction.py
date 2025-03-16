#%%
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
#%%

def predict_from_json(json_file_path='/Users/sayam_palrecha/my_project/Hackathon/test_data.json', model_path='/Users/sayam_palrecha/my_project/Hackathon/model.pkl'):
    """
    Load data from a JSON file and make predictions using the saved model

    Parameters:
    json_file_path (str): Path to the JSON file containing records to predict
    model_path (str): Path to the saved pickle model file

    Returns:
    DataFrame: Original data with added prediction columns
    """
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {json_file_path}")
    except FileNotFoundError:
        print(f"Error: JSON file {json_file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure all required features are present
    required_features = ['SpecifyTypeOfOperation', 'RecipientSignature',
                         'RaceAndEthnicDesignation', 'InterpreterSignature',
                         'DateOfRecipientSignature']

    # Check if all required columns are present
    missing_columns = [col for col in required_features if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in JSON data: {missing_columns}")
        return None

    # Create a copy of the dataframe for predictions
    df_with_predictions = df.copy()

    # Process the DateOfRecipientSignature to extract features needed by the model
    df['DateOfRecipientSignature'] = pd.to_datetime(df['DateOfRecipientSignature'])
    df['Month'] = df['DateOfRecipientSignature'].dt.month
    df['Year'] = df['DateOfRecipientSignature'].dt.year
    df['DayOfWeek'] = df['DateOfRecipientSignature'].dt.dayofweek

    # Prepare the features for prediction
    X_pred = df[['SpecifyTypeOfOperation', 'RecipientSignature', 'RaceAndEthnicDesignation',
                 'InterpreterSignature', 'Month', 'Year', 'DayOfWeek']]

    # Make predictions
    try:
        y_pred_proba = model.predict_proba(X_pred)[:, 1]
        y_pred_class = model.predict(X_pred)

        # Add predictions to the results dataframe
        df_with_predictions['RejectionProbability'] = y_pred_proba
        df_with_predictions['AcceptanceProbability'] = 1 - y_pred_proba
        df_with_predictions['PredictedOutcome'] = ['Rejected' if prob > 0.15 else 'Accepted' for prob in y_pred_proba]

        # Add risk scores (0-100 scale)
        df_with_predictions['RejectionRiskScore'] = (df_with_predictions['RejectionProbability'] * 100).round(1)
        df_with_predictions['AcceptanceRiskScore'] = (df_with_predictions['AcceptanceProbability'] * 100).round(1)

        # Add detailed interpretation
        def interpret_prediction(row):
            if row['RejectionProbability'] > 0.75:
                risk_level = "High risk of rejection"
                details = "Highly likely to be rejected due to non-compliance with 30-day waiting period"
            elif row['RejectionProbability'] > 0.5:
                risk_level = "Moderate risk of rejection"
                details = "May be rejected due to potential non-compliance issues"
            elif row['RejectionProbability'] > 0.25:
                risk_level = "Low risk of rejection"
                details = "Likely to be accepted but has some minor compliance concerns"
            else:
                risk_level = "Very low risk of rejection"
                details = "Highly likely to be accepted with strong compliance indicators"

            return f"{risk_level}: {details}"

        df_with_predictions['Interpretation'] = df_with_predictions.apply(interpret_prediction, axis=1)

        # Save the results to a new JSON file
        output_path = json_file_path.replace('.json', '_predictions.json')
        df_with_predictions.to_json(output_path, orient='records', indent=4)
        print(f"Predictions saved to {output_path}")

        # Save only rejected patients to a separate JSON file
        rejected_patients = df_with_predictions[df_with_predictions['PredictedOutcome'] == 'Rejected']
        if len(rejected_patients) > 0:
            rejected_output_path = json_file_path.replace('.json', '_rejected_patients.json')
            rejected_patients.to_json(rejected_output_path, orient='records', indent=4)
            print(f"Rejected patients saved to {rejected_output_path}")
            print(f"Number of rejected patients: {len(rejected_patients)} out of {len(df_with_predictions)}")
        else:
            print("No patients were predicted to be rejected.")

        # Print the results
        for i, row in df_with_predictions.iterrows():
            print(f"\nRecord {i + 1}:")
            print(f"  Operation: {row['SpecifyTypeOfOperation']}")
            print(f"  Race/Ethnicity: {row['RaceAndEthnicDesignation']}")
            print(f"  Date: {row['DateOfRecipientSignature']}")
            print(f"  Predicted Outcome: {row['PredictedOutcome']}")
            print(f"  Acceptance Score: {row['AcceptanceRiskScore']} / 100")
            print(f"  Rejection Score: {row['RejectionRiskScore']} / 100")
            print(f"  Interpretation: {row['Interpretation']}")

        return df_with_predictions

    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

#%%
if __name__ == "__main__":
    import sys

    # Filter out web server arguments like --host=127.0.0.1
    args = [arg for arg in sys.argv if not arg.startswith('--')]

    if len(args) > 1:
        json_path = args[1]
        model_path = args[2] if len(args) > 2 else 'model.pkl'
        predict_from_json(json_path, model_path)
    else:
        print("Usage: python json_prediction.py <json_file_path> [model_path]")
        print("Example: python json_prediction.py sample_prediction_data.json compliance_prediction_model.pkl")

        # If no arguments, try with the sample file
        try:
            predict_from_json('/Users/sayam_palrecha/my_project/Hackathon/test_data.json')
        except Exception as e:
            print(f"Error running with sample data: {e}")
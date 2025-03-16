#%%
import os
import sys
import pickle
import json
import pandas as pd
import importlib.util
import traceback

#%%
def import_module_from_file(file_path):
    """Import a module from a file path"""
    try:
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module from {file_path}: {e}")
        return None
#%%

def run_complete_pipeline(json_data_path='', model_path='', generate_data=True, train_model=True):
    """
    Run the complete TennCare compliance prediction pipeline

    Parameters:
    json_data_path (str): Path to the JSON file with records to predict
    model_path (str): Path to save/load the model
    generate_data (bool): Whether to generate synthetic data
    train_model (bool): Whether to train a new model

    Returns:
    DataFrame: The prediction results
    """
    # Set default paths if not provided
    if not model_path:
        model_path = 'compliance_prediction_model.pkl'

    if not json_data_path:
        json_data_path = 'sample_prediction_data.json'

    print("=" * 50)
    print("TennCare Compliance Prediction Pipeline")
    print("=" * 50)

    # Try to import modules from files in the current directory
    current_dir = os.getcwd()
    print(f"\nSearching for Python modules in: {current_dir}")

    # Find Python files in the current directory
    py_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != os.path.basename(__file__)]
    print(f"Found Python files: {py_files}")

    # Import modules
    modules = {}
    for py_file in py_files:
        file_path = os.path.join(current_dir, py_file)
        module = import_module_from_file(file_path)
        if module:
            modules[py_file.replace('.py', '')] = module
            print(f"Successfully imported module from {py_file}")
            # Print available functions in the module
            functions = [f for f in dir(module) if callable(getattr(module, f)) and not f.startswith('_')]
            print(f"  Available functions: {functions}")

    # 1. Check if we need to train a model
    if train_model or not os.path.exists(model_path):
        print(f"\nStep 1: Training model and saving to {model_path}")
        try:
            data_csv_path = 'tenncare_sterilization_consent_synthetic_data.csv'

            # Generate synthetic data if specified
            if generate_data and not os.path.exists(data_csv_path):
                print("Generating synthetic data...")

                # Look for data generation function in modules
                data_generator_found = False
                for module_name, module in modules.items():
                    if hasattr(module, 'generate_synthetic_data'):
                        print(f"Found generate_synthetic_data function in {module_name}")
                        df = module.generate_synthetic_data()
                        df.to_csv(data_csv_path, index=False)
                        print(f"Synthetic data generated and saved to {data_csv_path}")
                        data_generator_found = True
                        break

                if not data_generator_found:
                    print("Warning: Could not find data generation function. Please make sure the data CSV exists.")

            # Train the model
            model_trainer_found = False
            for module_name, module in modules.items():
                if hasattr(module, 'build_compliance_prediction_model'):
                    print(f"Found build_compliance_prediction_model function in {module_name}")
                    try:
                        model, _, _ = module.build_compliance_prediction_model(data_csv_path)
                        print(f"Model trained and saved to {model_path}")
                        model_trainer_found = True
                        break
                    except Exception as e:
                        print(f"Error training model: {e}")
                        traceback.print_exc()

            if not model_trainer_found:
                print("Error: Could not find model training function in any module.")
                return None

        except Exception as e:
            print(f"Error in model training step: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"\nStep 1: Using existing model from {model_path}")

    # 2. Make predictions on the JSON data
    print(f"\nStep 2: Making predictions on data from {json_data_path}")

    # Check if JSON file exists, create sample if it doesn't
    if not os.path.exists(json_data_path):
        print(f"JSON file {json_data_path} not found. Creating a sample file...")
        sample_data = [
            {
                "SpecifyTypeOfOperation": "Bilateral Tubal Ligation",
                "RecipientSignature": 1,
                "RaceAndEthnicDesignation": "Hispanic",
                "InterpreterSignature": 1,
                "DateOfRecipientSignature": "06/15/2023"
            },
            {
                "SpecifyTypeOfOperation": "Vasectomy",
                "RecipientSignature": 1,
                "RaceAndEthnicDesignation": "White",
                "InterpreterSignature": 0,
                "DateOfRecipientSignature": "03/22/2024"
            }
        ]

        with open(json_data_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
        print(f"Sample JSON file created at {json_data_path}")

    # Make predictions
    prediction_function_found = False
    for module_name, module in modules.items():
        if hasattr(module, 'predict_from_json'):
            print(f"Found predict_from_json function in {module_name}")
            try:
                results = module.predict_from_json(json_data_path, model_path)
                if results is None:
                    print("Error: No prediction results were returned.")
                    continue
                prediction_function_found = True
                break
            except Exception as e:
                print(f"Error making predictions with {module_name}.predict_from_json: {e}")
                traceback.print_exc()

    if not prediction_function_found:
        print("Error: Could not find prediction function in any module.")
        return None

    # 3. Visualize the results if visualization is available
    prediction_json_path = json_data_path.replace('.json', '_predictions.json')

    visualization_found = False
    for module_name, module in modules.items():
        if hasattr(module, 'visualize_predictions'):
            print(f"Found visualize_predictions function in {module_name}")
            try:
                module.visualize_predictions(prediction_json_path)
                visualization_found = True
                break
            except Exception as e:
                print(f"Error visualizing predictions: {e}")
                traceback.print_exc()

    if not visualization_found:
        print("\nVisualization not available - skipping visualization step")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)

    return results

#%%
if __name__ == "__main__":
    # Get command line args (ignoring web server args like --host)
    args = [arg for arg in sys.argv if not arg.startswith('--')]

    # Get optional JSON file path
    if len(args) > 1:
        json_path = args[1]
    else:
        json_path = 'sample_prediction_data.json'

    # Run the pipeline
    run_complete_pipeline(json_data_path=json_path)
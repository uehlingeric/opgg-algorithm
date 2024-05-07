import pandas as pd
from tabulate import tabulate

import sys
sys.path.append('../')
from models import model

def load_test_data(filepath):
    return pd.read_csv(filepath)

def evaluate_predictions(predictions, actuals):
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    return mse, r2

def print_detailed_results(predictions, actuals, data):
    results = []
    headers = ["Predicted OP Score", "Actual OP Score", "Win", "Role", "Champ", "Length", "KDA"]
    for pred, actual, win, role, champ, length, kda in zip(predictions, actuals, data['win'], data['position'], data['champ'], data['length'], data['kda']):
        results.append([f"{pred:.3f}", f"{actual:.3f}", win, role, champ, f"{length:.3f}", f"{kda:.3f}"])
    print(tabulate(results, headers=headers, tablefmt="pretty"))

if __name__ == "__main__":
    test_data = load_test_data('../data/processed/test.csv')
    actual_scores = test_data['op_score']
    test_data_for_printing = test_data[['win', 'position', 'champ', 'length', 'kda']].copy()
    test_data = test_data.drop(columns=['op_score'])
    
    preprocessor, keras_model = model.load_assets('../models/preprocessor.pkl', '../models/advanced_model.h5')
    predictions = model.make_predictions(test_data, preprocessor, keras_model)
    mse, r2 = evaluate_predictions(predictions.flatten(), actual_scores)
    print_detailed_results(predictions.flatten(), actual_scores, test_data_for_printing)
    
    print("\nGlobal Metrics:")
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

from src.preprocessing import load_data
from src.train import train_models, save_model
from src.evaluate import evaluate_models, select_best_by_r2, plot_rf_feature_importance
import argparse
import os
import json
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--csvs", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    paths = []
    if args.csvs:
        paths = [p.strip() for p in args.csvs.split(",") if p.strip()]
    elif args.csv:
        paths = [args.csv]
    else:
        paths = [None]
    for p in paths:
        try:
            X, y = load_data(p)
        except (FileNotFoundError, KeyError) as err:
            print(err)
            continue
        lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(X, y, tune_rf=not args.fast)
        results = evaluate_models(lr_model, dt_model, rf_model, X_test, y_test)
        best_name, best_model = select_best_by_r2(lr_model, dt_model, rf_model, X_test, y_test)
        if best_name == "RandomForestRegressor":
            plot_rf_feature_importance(rf_model, X_test.columns)
        stem = "auto" if p is None else os.path.splitext(os.path.basename(p))[0]
        model_fname = f"model_{stem}.pkl"
        cols_fname = f"feature_columns_{stem}.json"
        save_model(best_model, X_train.columns, filename=model_fname, columns_filename=cols_fname)
        os.makedirs("models", exist_ok=True)
        with open(os.path.join("models", f"evaluation_{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(results.to_dict(orient="records"), f)
 
 
if __name__ == "__main__":
     main()

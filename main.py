 from src.preprocessing import load_data
 from src.train import train_models, save_model
 from src.evaluate import evaluate_models, select_best_by_r2, plot_rf_feature_importance
 
 
 def main():
     X, y = load_data()
     lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(X, y)
     results = evaluate_models(lr_model, dt_model, rf_model, X_test, y_test)
     best_name, best_model = select_best_by_r2(lr_model, dt_model, rf_model, X_test, y_test)
     if best_name == "RandomForestRegressor":
         plot_rf_feature_importance(rf_model, X_test.columns)
     save_model(best_model, X_train.columns)
 
 
 if __name__ == "__main__":
     main()

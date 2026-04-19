from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def evaluate_on_test(model, X_test, y_test):
    # Оценка модели на тестовой выборке
    y_pred_test = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("Результаты на тесте")
    print(f"MAE: {test_mae:.2f} циклов")
    print(f"RMSE: {test_rmse:.2f} циклов")
    print(f"R2: {test_r2:.3f}")
    
    return test_mae, test_rmse, test_r2
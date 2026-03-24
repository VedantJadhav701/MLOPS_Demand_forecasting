import numpy as np

def track_performance(y_true, y_pred):
    """
    Simulates live performance monitoring.
    Compares live predictions vs actuals.
    """
    mape = (abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100
    
    print(f"[LIVE MONITOR] MAPE: {mape:.2f}%")
    
    return mape

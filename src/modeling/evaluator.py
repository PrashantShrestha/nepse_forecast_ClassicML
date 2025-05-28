from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,
    classification_report
)
import pandas as pd
import json
import os
import numpy as np
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('evaluator')

def evaluate_model(model, le, X_test, y_test):
    """Calculate evaluation metrics"""
    if X_test is None or y_test is None:
        return None
        
    y_pred = model.predict(X_test)
    
    # Decode labels
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_test_decoded, y_pred_decoded),
        'precision': precision_score(y_test_decoded, y_pred_decoded, average='weighted'),
        'recall': recall_score(y_test_decoded, y_pred_decoded, average='weighted'),
        'f1': f1_score(y_test_decoded, y_pred_decoded, average='weighted')
    }
    
    # Classification report
    report = classification_report(
        y_test_decoded, 
        y_pred_decoded,
        output_dict=True,
        zero_division=0
    )
    metrics['class_report'] = report
    
    # Confusion matrix
    classes = le.classes_
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=classes)
    metrics['confusion_matrix'] = {
        'labels': classes.tolist(),
        'matrix': cm.tolist()
    }
    
    logger.info(f"Model evaluation completed: Accuracy={metrics['accuracy']:.2f}")
    return metrics

def log_evaluation(metrics, training_time, horizon):
    """Save evaluation results to logs"""
    if metrics is None:
        return
        
    log_entry = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'horizon': horizon,
        'broker_mode': config['training']['broker_mode'],
        'training_time_sec': training_time,
        **metrics
    }
    
    # Save to JSON
    log_dir = config['logs']['log_dir']
    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    json_path = os.path.join(log_dir, f"eval_{date_str}.json")
    
    with open(json_path, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # Save to CSV
    csv_path = os.path.join(log_dir, "evaluation_history.csv")
    try:
        history = pd.read_csv(csv_path)
    except FileNotFoundError:
        history = pd.DataFrame()
    
    # Flatten class report
    report_df = pd.DataFrame(metrics['class_report']).transpose().reset_index()
    report_df['timestamp'] = log_entry['timestamp']
    
    # Main metrics
    flat_entry = {
        'date': date_str,
        'horizon': horizon,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'training_time': training_time
    }
    
    history = pd.concat([history, pd.DataFrame([flat_entry])], ignore_index=True)
    history.to_csv(csv_path, index=False)
    
    # Save classification report
    report_path = os.path.join(log_dir, f"class_report_{date_str}.csv")
    report_df.to_csv(report_path, index=False)
    
    logger.info(f"Evaluation results logged to {json_path} and {csv_path}")
    return history
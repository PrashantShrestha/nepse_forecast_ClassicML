import time
import pandas as pd
import os
from datetime import datetime, timedelta
from src.processing.data_loader import load_raw_data
from src.processing.feature_engineering import calculate_technical_features, calculate_broker_features
from src.processing.target_generator import generate_targets
from src.modeling.trainer import IncrementalTrainer
from src.modeling.evaluator import evaluate_model
from src.utils.data_manager import save_features, save_targets, save_model
from src.utils.config_loader import load_config
from src.utils.logger import log_evaluation, setup_logger

config = load_config()
logger = setup_logger('pipeline', 'pipeline.log')

def run_pipeline():
    logger.info("Starting daily pipeline")
    start_time = time.time()
    
    # 1. Data loading and cleaning
    logger.info("Loading raw data")
    raw_data = load_raw_data()
    
    # 2. Feature engineering
    logger.info("Calculating technical features")
    tech_features = calculate_technical_features(raw_data)
    save_features(tech_features, 'technical')
    
    logger.info(f"Calculating broker features ({config['training']['broker_mode']} mode)")
    broker_features = calculate_broker_features(raw_data, mode=config['training']['broker_mode'])
    save_features(broker_features, f"broker_{config['training']['broker_mode']}")
    
    # 3. Target generation
    logger.info("Generating targets")
    targets = generate_targets(tech_features[['Date', 'Symbol', 'Rate']])
    save_targets(targets)
    
    # 4. Model training
    logger.info("Starting model training")
    trainer = IncrementalTrainer()
    X_test, y_test = trainer.train()
    training_duration = time.time() - start_time
    
    # 5. Model evaluation
    logger.info("Evaluating model")
    metrics = evaluate_model(trainer.model, X_test, y_test)
    metrics['training_time'] = training_duration
    
    # 6. Save model and logs
    current_date = datetime.now().strftime('%Y%m%d')
    model_path = os.path.join(
        config['models']['base_path'],
        'random_forest',
        config['training']['horizon'],
        f"model_{current_date}.pkl"
    )
    
    metadata = {
        'training_date': current_date,
        'horizon': config['training']['horizon'],
        'broker_mode': config['training']['broker_mode'],
        'feature_columns': X_test.columns.tolist(),
        'metrics': metrics,
        'training_duration': training_duration
    }
    
    save_model(trainer.model, model_path, metadata)
    log_evaluation(current_date, metrics, config['training']['horizon'])
    
    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_pipeline()
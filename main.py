# main.py - DAILY PIPELINE
import time
import pandas as pd
from datetime import datetime
from src.processing.data_loader import load_raw_data
from src.processing.feature_engineering import calculate_technical_features, calculate_broker_features
from src.processing.target_generator import generate_targets
from src.modeling.trainer import IncrementalTrainer
from src.modeling.evaluator import evaluate_model, log_evaluation
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

def run_pipeline():
    config = load_config()
    logger = get_logger('pipeline', config['logs']['log_dir'])
    logger.info("Starting daily pipeline")
    
    
    # 1. Data loading and cleaning
    logger.info("Loading raw data")
    raw_data = load_raw_data()
    
    # 2. Feature engineering
    logger.info("Calculating technical features")
    tech_features = calculate_technical_features(raw_data)
    
    logger.info(f"Calculating broker features ({config['training']['broker_mode']} mode)")
    broker_features = calculate_broker_features(raw_data, mode=config['training']['broker_mode'])
    
    # 3. Target generation
    logger.info("Generating targets")
    targets = generate_targets(tech_features)
    
    # 4. Model training
    logger.info("Starting model training")
    trainer = IncrementalTrainer()
    X_test, y_test, training_duration = trainer.train(tech_features, broker_features, targets)
    
    # 5. Model evaluation
    if X_test is not None and y_test is not None:
        logger.info("Evaluating model")
        metrics = evaluate_model(trainer.model, trainer.le, X_test, y_test)
        log_evaluation(metrics, training_duration, config['training']['horizon'])
    
    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_pipeline()
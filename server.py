from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import os
from pair_trading_model import PairTradingModel  # Import our model from the previous file

app = Flask(__name__, static_folder='static')

# Store active models in memory
active_models = {}

@app.route('/api/test_connection', methods=['GET'])
def test_connection():
    """Simple endpoint to test if the API is running"""
    return jsonify({"status": "success", "message": "API is running"})

@app.route('/api/available_tickers', methods=['GET'])
def available_tickers():
    """Return a list of popular stock tickers for the frontend dropdown"""
    # This is a simple implementation - in a production app, you might want to
    # fetch this dynamically or store a more comprehensive list
    popular_tickers = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "V", "name": "Visa Inc."},
        {"symbol": "JNJ", "name": "Johnson & Johnson"},
        {"symbol": "WMT", "name": "Walmart Inc."},
        {"symbol": "PG", "name": "Procter & Gamble Co."},
        {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
        {"symbol": "BAC", "name": "Bank of America Corp."},
        {"symbol": "KO", "name": "The Coca-Cola Company"},
        {"symbol": "DIS", "name": "The Walt Disney Company"},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "PFE", "name": "Pfizer Inc."},
        {"symbol": "CSCO", "name": "Cisco Systems Inc."},
        {"symbol": "INTC", "name": "Intel Corporation"}
    ]
    return jsonify(popular_tickers)

@app.route('/api/validate_pair', methods=['POST'])
def validate_pair():
    """Validate a pair of tickers by checking correlation"""
    data = request.get_json()
    ticker1 = data.get('ticker1')
    ticker2 = data.get('ticker2')
    
    if not ticker1 or not ticker2:
        return jsonify({"status": "error", "message": "Both tickers are required"}), 400
    
    try:
        # Create a temporary model to fetch data and calculate correlation
        model = PairTradingModel(ticker1, ticker2)
        data = model.fetch_data(
            start_date=(datetime.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        )
        
        # Calculate correlation
        correlation = data[ticker1].corr(data[ticker2])
        
        # Check if correlation is strong enough
        is_valid = abs(correlation) > 0.7  # Arbitrary threshold
        
        return jsonify({
            "status": "success",
            "correlation": correlation,
            "is_valid": bool(is_valid),
            "message": f"Correlation: {correlation:.2f}" + (
                " - Good pair candidate!" if is_valid else " - Weak correlation, consider another pair."
            )
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/create_model', methods=['POST'])
def create_model():
    """Create and run a pair trading model with the given parameters"""
    data = request.get_json()
    ticker1 = data.get('ticker1')
    ticker2 = data.get('ticker2')
    lookback_period = int(data.get('lookback_period', 60))
    z_threshold = float(data.get('z_threshold', 2.0))
    start_date = data.get('start_date')
    
    if not ticker1 or not ticker2:
        return jsonify({"status": "error", "message": "Both tickers are required"}), 400
    
    try:
        # Create a new model with the specified parameters
        model = PairTradingModel(ticker1, ticker2, lookback_period, z_threshold)
        
        # Fetch data and generate signals
        model.fetch_data(start_date=start_date)
        model.calculate_hedge_ratio(method='rolling')
        model.calculate_spread()
        model.calculate_z_scores()
        model.generate_signals()
        performance, metrics = model.backtest()
        
        # Create a unique model ID
        model_id = f"{ticker1}_{ticker2}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store the model in memory
        active_models[model_id] = model
        
        # Prepare data for the frontend
        result_data = {
            "model_id": model_id,
            "tickers": {
                "ticker1": ticker1,
                "ticker2": ticker2
            },
            "parameters": {
                "lookback_period": lookback_period,
                "z_threshold": z_threshold
            },
            "metrics": {
                "total_return": float(metrics['total_return']),
                "annual_return": float(metrics['annual_return']),
                "sharpe_ratio": float(metrics['sharpe_ratio']),
                "max_drawdown": float(metrics['max_drawdown'])
            },
            "data_summary": {
                "start_date": model.data.index[0].strftime('%Y-%m-%d'),
                "end_date": model.data.index[-1].strftime('%Y-%m-%d'),
                "num_trading_days": len(model.data)
            }
        }
        
        return jsonify({"status": "success", "model": result_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get_model_data/<model_id>', methods=['GET'])
def get_model_data(model_id):
    """Get detailed data for a specific model"""
    if model_id not in active_models:
        return jsonify({"status": "error", "message": "Model not found"}), 404
    
    model = active_models[model_id]
    
    # Convert data to JSON-serializable format
    price_data = []
    for date, row in model.data.iterrows():
        price_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "ticker1_price": float(row[model.ticker1]),
            "ticker2_price": float(row[model.ticker2])
        })
    
    spread_data = []
    for date, value in model.spread.items():
        if date in model.z_scores.index:
            spread_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "spread": float(value),
                "z_score": float(model.z_scores[date]) if not np.isnan(model.z_scores[date]) else None
            })
    
    signal_data = []
    for date, row in model.signals.iterrows():
        if not pd.isna(row['signal']):
            signal_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "signal": int(row['signal'])
            })
    
    performance_data = []
    for date, row in model.performance.iterrows():
        if not pd.isna(row['portfolio_value']):
            performance_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "portfolio_value": float(row['portfolio_value']),
                "drawdown": float(row['drawdown']) if not np.isnan(row['drawdown']) else None
            })
    
    return jsonify({
        "status": "success",
        "model_id": model_id,
        "ticker1": model.ticker1,
        "ticker2": model.ticker2,
        "price_data": price_data,
        "spread_data": spread_data,
        "signal_data": signal_data,
        "performance_data": performance_data
    })

@app.route('/api/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model from memory"""
    if model_id in active_models:
        del active_models[model_id]
        return jsonify({"status": "success", "message": f"Model {model_id} deleted"})
    else:
        return jsonify({"status": "error", "message": "Model not found"}), 404

# CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)
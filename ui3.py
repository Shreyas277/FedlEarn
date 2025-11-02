from flask import Flask, render_template_string, send_file, request, jsonify
import io
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Insights - ML Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0e27;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated Background Pattern */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(88, 86, 214, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(255, 107, 107, 0.2) 0%, transparent 50%);
            animation: backgroundShift 15s ease infinite;
            z-index: 0;
        }
        
        @keyframes backgroundShift {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(50px, 30px) scale(1.1); }
        }
        
        /* Geometric Pattern Overlay */
        body::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(30deg, rgba(255,255,255,0.02) 12%, transparent 12.5%, transparent 87%, rgba(255,255,255,0.02) 87.5%, rgba(255,255,255,0.02)),
                linear-gradient(150deg, rgba(255,255,255,0.02) 12%, transparent 12.5%, transparent 87%, rgba(255,255,255,0.02) 87.5%, rgba(255,255,255,0.02)),
                linear-gradient(30deg, rgba(255,255,255,0.02) 12%, transparent 12.5%, transparent 87%, rgba(255,255,255,0.02) 87.5%, rgba(255,255,255,0.02)),
                linear-gradient(150deg, rgba(255,255,255,0.02) 12%, transparent 12.5%, transparent 87%, rgba(255,255,255,0.02) 87.5%, rgba(255,255,255,0.02)),
                linear-gradient(60deg, rgba(255,255,255,0.05) 25%, transparent 25.5%, transparent 75%, rgba(255,255,255,0.05) 75%, rgba(255,255,255,0.05)),
                linear-gradient(60deg, rgba(255,255,255,0.05) 25%, transparent 25.5%, transparent 75%, rgba(255,255,255,0.05) 75%, rgba(255,255,255,0.05));
            background-size: 80px 140px;
            background-position: 0 0, 0 0, 40px 70px, 40px 70px, 0 0, 40px 70px;
            z-index: 0;
        }
        
        .container {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 
                0 0 80px rgba(120, 119, 198, 0.3),
                0 30px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            padding: 60px;
            max-width: 1000px;
            width: 100%;
            animation: fadeIn 0.6s ease-in;
            position: relative;
            z-index: 1;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .logo {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .logo-text {
            font-size: 2.8em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }
        
        .tagline {
            color: #94a3b8;
            text-align: center;
            margin-bottom: 50px;
            font-size: 1.1em;
            font-weight: 300;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        .main-screen {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 40px;
            justify-content: center;
        }
        
        .btn-main {
            padding: 50px 40px;
            font-size: 1.6em;
            font-weight: 600;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            color: white;
            text-transform: uppercase;
            letter-spacing: 3px;
            position: relative;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
        }
        
        .btn-main::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .btn-main:hover::before {
            left: 100%;
        }
        
        .btn-icon {
            font-size: 2.5em;
            display: block;
            margin-bottom: 15px;
        }
        
        .btn-train {
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
            border-color: rgba(245, 87, 108, 0.3);
        }
        
        .btn-train:hover {
            transform: translateY(-8px);
            box-shadow: 
                0 20px 40px rgba(245, 87, 108, 0.4),
                0 0 20px rgba(245, 87, 108, 0.3);
            border-color: rgba(245, 87, 108, 0.6);
            background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%);
        }
        
        .btn-use {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
            border-color: rgba(79, 172, 254, 0.3);
        }
        
        .btn-use:hover {
            transform: translateY(-8px);
            box-shadow: 
                0 20px 40px rgba(79, 172, 254, 0.4),
                0 0 20px rgba(79, 172, 254, 0.3);
            border-color: rgba(79, 172, 254, 0.6);
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.2) 100%);
        }
        
        .hidden {
            display: none;
        }
        
        .train-screen, .use-screen {
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        h2 {
            color: #e2e8f0;
            margin-bottom: 35px;
            font-size: 2em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        h2::before {
            content: '';
            width: 5px;
            height: 40px;
            background: linear-gradient(180deg, #667eea, #764ba2);
            border-radius: 10px;
        }
        
        .instructions {
            background: rgba(30, 41, 59, 0.5);
            padding: 35px;
            border-radius: 16px;
            margin-bottom: 35px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .instruction-item {
            padding: 20px 25px;
            margin: 15px 0;
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #667eea;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            color: #cbd5e1;
        }
        
        .instruction-item:hover {
            transform: translateX(8px);
            border-left-color: #764ba2;
            background: rgba(15, 23, 42, 0.8);
        }
        
        .instruction-item strong {
            color: #a78bfa;
            font-size: 1.1em;
            font-weight: 600;
            display: block;
            margin-bottom: 8px;
        }
        
        .btn-download {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 18px 45px;
            border: none;
            border-radius: 12px;
            font-size: 1.15em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
            display: block;
            margin: 0 auto;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-download:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 35px rgba(16, 185, 129, 0.6);
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
        }
        
        .form-group {
            margin: 25px 0;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 10px;
            color: #e2e8f0;
            font-weight: 500;
            font-size: 1em;
            letter-spacing: 0.5px;
        }
        
        .form-group input {
            width: 100%;
            padding: 16px 20px;
            background: rgba(30, 41, 59, 0.6);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-size: 1em;
            transition: all 0.3s ease;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(30, 41, 59, 0.8);
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .form-group input::placeholder {
            color: #64748b;
        }
        
        .btn-predict {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 18px 45px;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4);
            width: 100%;
            margin-top: 30px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-predict:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 35px rgba(245, 158, 11, 0.6);
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        }
        
        .btn-back {
            background: rgba(71, 85, 105, 0.5);
            color: #e2e8f0;
            padding: 12px 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            margin-bottom: 25px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .btn-back:hover {
            background: rgba(71, 85, 105, 0.8);
            transform: translateX(-5px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .result {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            border: 2px solid rgba(102, 126, 234, 0.4);
            color: #e2e8f0;
            padding: 35px;
            border-radius: 16px;
            margin-top: 35px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 600;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
            animation: resultPulse 0.6s ease-in;
            backdrop-filter: blur(10px);
        }
        
        @keyframes resultPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.03); }
        }
        
        .result-label {
            font-size: 0.7em;
            color: #a78bfa;
            display: block;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .result-value {
            font-size: 1.8em;
            color: #fff;
            text-shadow: 0 2px 10px rgba(102, 126, 234, 0.5);
        }
        
        /* Floating particles animation */
        .particle {
            position: fixed;
            width: 4px;
            height: 4px;
            background: rgba(102, 126, 234, 0.6);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
            animation: float 15s infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) translateX(0); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-100vh) translateX(50px); opacity: 0; }
        }
    </style>
</head>
<body>
    <!-- Floating particles -->
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 40%; animation-delay: 6s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 60%; animation-delay: 10s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
    <div class="particle" style="left: 80%; animation-delay: 14s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 3s;"></div>
    
    <div class="container">
        <div class="logo">
            <div class="logo-text">Federated Learning Insights</div>
        </div>
        <div class="tagline">PLATFORM TO TRAIN MODEL ON THE PRIVATE DATA WITH PRIVACY</div>
        
        <!-- Main Screen -->
        <div id="mainScreen" class="main-screen">
            <button class="btn-main btn-train" onclick="showTrain()">
                <span class="btn-icon">🧠</span>
                <span style="position: relative; z-index: 1;">Train</span>
            </button>
            <button class="btn-main btn-use" onclick="showUse()">
                <span class="btn-icon">⚡</span>
                <span style="position: relative; z-index: 1;">Use</span>
            </button>
        </div>
        
        <!-- Train Screen -->
        <div id="trainScreen" class="hidden train-screen">
            <button class="btn-back" onclick="showMain()">← Back to Home</button>
            <h2>Training Guidelines</h2>
            
            <div class="instructions">
                <div class="instruction-item">
                    <strong>Step 1: Data Preparation</strong>
                    Organize your dataset with all required attributes in a structured format (CSV, Excel, or JSON)
                </div>
                <div class="instruction-item">
                    <strong>Step 2: Target Variable Validation</strong>
                    Ensure the primary regression output column 'delivery_risk' exists and contains numeric values only
                </div>
                <div class="instruction-item">
                    <strong>Step 3: Numerical Features Handling</strong>
                    Verify all 15 numerical features (late_delivery_risk, order_item_profit_ratio, order_item_quantity, order_year, order_month, order_day, order_hour, order_minute, order_weekend, ship_year, ship_month, ship_day, ship_hour, ship_minute, ship_weekend) have no missing (NaN) values. Impute with mean/median if necessary
                </div>
                <div class="instruction-item">
                    <strong>Step 4: Categorical Features Consistency</strong>
                    Confirm all 5 nominal features (delivery_status, customer_segment, department_name, order_status, shipping_mode) exist and their values match the hardcoded NOMINAL_CATEGORIES list exactly
                </div>
                <div class="instruction-item">
                    <strong>Step 5: Feature Engineering Cleanup</strong>
                    Ensure all date/time columns are decomposed into 10 separate numeric features (order_year, order_month, order_day, order_hour, order_minute, order_weekend, ship_year, ship_month, ship_day, ship_hour, ship_minute, ship_weekend) as required by the model
                </div>
                <div class="instruction-item">
                    <strong>Step 6: Data Type Validation</strong>
                    Validate that all 15 numerical features are stored as Python-readable numeric types (integer or float) and all 20 features plus 1 target variable are convertible to their expected types
                </div>
                <div class="instruction-item">
                    <strong>Step 7: Feature Vector Size Validation</strong>
                    After preprocessing (StandardScaling for 15 numeric features and OneHotEncoding for 5 nominal features), verify the final feature count is exactly 41 across all files for model compatibility
                </div>
            </div>
            
            <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
                <button class="btn-download" onclick="downloadTemplate()">
                    📥 Download Training Template
                </button>
                <button class="btn-download" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);" onclick="downloadClient()">
                    📄 Download Client.py
                </button>
            </div>
        </div>
        
        <!-- Use Screen -->
        <div id="useScreen" class="hidden use-screen">
            <button class="btn-back" onclick="showMain()">← Back to Home</button>
            <h2>Model Prediction Interface</h2>
            
            <form id="predictionForm">
                <h3 style="color: #a78bfa; margin: 25px 0 15px 0; font-size: 1.3em;">Numerical Features</h3>
                
                <div class="form-group">
                    <label>Late Delivery Risk</label>
                    <input type="number" step="any" name="late_delivery_risk" placeholder="e.g., 1" required>
                </div>
                <div class="form-group">
                    <label>Order Item Profit Ratio</label>
                    <input type="number" step="any" name="order_item_profit_ratio" placeholder="e.g., 0.28" required>
                </div>
                <div class="form-group">
                    <label>Order Item Quantity</label>
                    <input type="number" step="any" name="order_item_quantity" placeholder="e.g., 7" required>
                </div>
                <div class="form-group">
                    <label>Order Year</label>
                    <input type="number" name="order_year" placeholder="e.g., 2024" required>
                </div>
                <div class="form-group">
                    <label>Order Month</label>
                    <input type="number" name="order_month" min="1" max="12" placeholder="1-12" required>
                </div>
                <div class="form-group">
                    <label>Order Day</label>
                    <input type="number" name="order_day" min="1" max="31" placeholder="1-31" required>
                </div>
                <div class="form-group">
                    <label>Order Hour</label>
                    <input type="number" name="order_hour" min="0" max="23" placeholder="0-23" required>
                </div>
                <div class="form-group">
                    <label>Order Minute</label>
                    <input type="number" name="order_minute" min="0" max="59" placeholder="0-59" required>
                </div>
                <div class="form-group">
                    <label>Order Weekend</label>
                    <select name="order_weekend" required style="width: 100%; padding: 16px 20px; background: rgba(30, 41, 59, 0.6); border: 2px solid rgba(255, 255, 255, 0.1); border-radius: 10px; font-size: 1em; color: #e2e8f0; font-family: 'Inter', sans-serif;">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Ship Year</label>
                    <input type="number" name="ship_year" placeholder="e.g., 2024" required>
                </div>
                <div class="form-group">
                    <label>Ship Month</label>
                    <input type="number" name="ship_month" min="1" max="12" placeholder="1-12" required>
                </div>
                <div class="form-group">
                    <label>Ship Day</label>
                    <input type="number" name="ship_day" min="1" max="31" placeholder="1-31" required>
                </div>
                <div class="form-group">
                    <label>Ship Hour</label>
                    <input type="number" name="ship_hour" min="0" max="23" placeholder="0-23" required>
                </div>
                <div class="form-group">
                    <label>Ship Minute</label>
                    <input type="number" name="ship_minute" min="0" max="59" placeholder="0-59" required>
                </div>
                <div class="form-group">
                    <label>Ship Weekend</label>
                    <select name="ship_weekend" required style="width: 100%; padding: 16px 20px; background: rgba(30, 41, 59, 0.6); border: 2px solid rgba(255, 255, 255, 0.1); border-radius: 10px; font-size: 1em; color: #e2e8f0; font-family: 'Inter', sans-serif;">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
                
                <h3 style="color: #a78bfa; margin: 35px 0 15px 0; font-size: 1.3em;">Categorical Features</h3>
                
                <div class="form-group">
                    <label>Delivery Status</label>
                    <input type="text" name="delivery_status" placeholder="e.g., Late delivery, On Time" required>
                </div>
                <div class="form-group">
                    <label>Customer Segment</label>
                    <input type="text" name="customer_segment" placeholder="e.g., Consumer, Corporate, Home Office" required>
                </div>
                <div class="form-group">
                    <label>Department Name</label>
                    <input type="text" name="department_name" placeholder="e.g., Golf, Electronics, Furniture" required>
                </div>
                <div class="form-group">
                    <label>Order Status</label>
                    <input type="text" name="order_status" placeholder="e.g., PROCESSING, COMPLETE, PENDING" required>
                </div>
                <div class="form-group">
                    <label>Shipping Mode</label>
                    <input type="text" name="shipping_mode" placeholder="e.g., Standard Class, Express, Same Day" required>
                </div>
                
                <button type="submit" class="btn-predict">🎯 Generate Prediction</button>
            </form>
            
            <div id="result" class="hidden result">
                <span class="result-label">Predicted Delivery Risk</span>
                <div class="result-value"></div>
            </div>
        </div>
    </div>
    
    <script>
        function showMain() {
            document.getElementById('mainScreen').classList.remove('hidden');
            document.getElementById('trainScreen').classList.add('hidden');
            document.getElementById('useScreen').classList.add('hidden');
        }
        
        function showTrain() {
            document.getElementById('mainScreen').classList.add('hidden');
            document.getElementById('trainScreen').classList.remove('hidden');
            document.getElementById('useScreen').classList.add('hidden');
        }
        
        function showUse() {
            document.getElementById('mainScreen').classList.add('hidden');
            document.getElementById('trainScreen').classList.add('hidden');
            document.getElementById('useScreen').classList.remove('hidden');
        }
        
        function downloadTemplate() {
            window.location.href = '/download/template';
        }
        
        function downloadClient() {
            window.location.href = '/download/client';
        }
        
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            
            // Collect all form data
            formData.forEach((value, key) => {
                // Convert numeric fields to appropriate types
                if (['late_delivery_risk', 'order_item_profit_ratio', 'order_item_quantity',
                     'order_year', 'order_month', 'order_day', 'order_hour', 'order_minute',
                     'order_weekend', 'ship_year', 'ship_month', 'ship_day', 'ship_hour',
                     'ship_minute', 'ship_weekend'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            });
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                const resultValue = resultDiv.querySelector('.result-value');
                
                if (response.ok) {
                    resultValue.textContent = result.prediction;
                    resultDiv.classList.remove('hidden');
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    resultValue.textContent = `Error: ${result.message || 'Prediction failed'}`;
                    resultDiv.classList.remove('hidden');
                    resultDiv.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%)';
                    resultDiv.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                }
            } catch (error) {
                const resultDiv = document.getElementById('result');
                const resultValue = resultDiv.querySelector('.result-value');
                resultValue.textContent = `Error: ${error.message}`;
                resultDiv.classList.remove('hidden');
                resultDiv.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%)';
                resultDiv.style.borderColor = 'rgba(239, 68, 68, 0.4)';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/download/template')
def download_template():
    # Create a professional training template file with all features
    template_content = """═══════════════════════════════════════════════════════════
         FEDERATED INSIGHTS - ML TRAINING TEMPLATE
         DELIVERY RISK PREDICTION MODEL
═══════════════════════════════════════════════════════════

DATASET REQUIREMENTS:
────────────────────────────────────────────────────────────

1. TARGET VARIABLE (1 column)
   ─────────────────────────────────────────────────────────
   Column Name: delivery_risk
   Type: Numeric (float or integer)
   Description: Primary regression output for delivery risk prediction

2. NUMERICAL FEATURES (15 columns - REQUIRED)
   ─────────────────────────────────────────────────────────
   - late_delivery_risk (numeric)
   - order_item_profit_ratio (numeric)
   - order_item_quantity (numeric)
   - order_year (integer)
   - order_month (integer: 1-12)
   - order_day (integer: 1-31)
   - order_hour (integer: 0-23)
   - order_minute (integer: 0-59)
   - order_weekend (binary: 0 or 1)
   - ship_year (integer)
   - ship_month (integer: 1-12)
   - ship_day (integer: 1-31)
   - ship_hour (integer: 0-23)
   - ship_minute (integer: 0-59)
   - ship_weekend (binary: 0 or 1)

   NOTE: All numerical features must be complete (no NaN/missing values)
         If missing values exist, impute with mean/median before training

3. CATEGORICAL FEATURES (5 columns - REQUIRED)
   ─────────────────────────────────────────────────────────
   Column Name: delivery_status
   Valid Values: Must match hardcoded NOMINAL_CATEGORIES exactly
   
   Column Name: customer_segment
   Valid Values: Must match hardcoded NOMINAL_CATEGORIES exactly
   
   Column Name: department_name
   Valid Values: Must match hardcoded NOMINAL_CATEGORIES exactly
   
   Column Name: order_status
   Valid Values: Must match hardcoded NOMINAL_CATEGORIES exactly
   
   Column Name: shipping_mode
   Valid Values: Must match hardcoded NOMINAL_CATEGORIES exactly

4. CSV STRUCTURE EXAMPLE
   ─────────────────────────────────────────────────────────
delivery_risk,late_delivery_risk,order_item_profit_ratio,order_item_quantity,order_year,order_month,order_day,order_hour,order_minute,order_weekend,ship_year,ship_month,ship_day,ship_hour,ship_minute,ship_weekend,delivery_status,customer_segment,department_name,order_status,shipping_mode
2.5,0.8,1.25,3,2024,1,15,10,30,0,2024,1,17,14,45,0,On Time,Corporate,Electronics,Complete,Standard
1.2,0.3,0.95,1,2024,2,20,8,15,1,2024,2,22,9,30,1,Late,Consumer,Furniture,Pending,Express
3.1,0.9,1.50,5,2024,3,10,14,0,0,2024,3,12,16,20,0,On Time,Home Office,Apparel,Complete,Same Day
   ─────────────────────────────────────────────────────────

5. DATA VALIDATION CHECKLIST
   ─────────────────────────────────────────────────────────
   ☐ All 21 columns present (1 target + 15 numeric + 5 categorical)
   ☐ No missing values in numerical features (impute if needed)
   ☐ All categorical values match NOMINAL_CATEGORIES
   ☐ Date/time features properly decomposed (no raw datetime columns)
   ☐ Numerical columns stored as numeric types (int/float)
   ☐ File format: CSV with UTF-8 encoding
   ☐ Minimum 100 rows recommended for training
     
6. FEATURE IMPORTANCE REFERENCE
   ─────────────────────────────────────────────────────────
   High Impact Features (typically):
   - late_delivery_risk
   - order_item_profit_ratio
   - shipping_mode
   - delivery_status
   
   Time-based Features:
   - order_year, order_month, order_day, order_hour, order_minute
   - ship_year, ship_month, ship_day, ship_hour, ship_minute
   - order_weekend, ship_weekend

═══════════════════════════════════════════════════════════
For support: support@neuralinsights.ai
Documentation: https://docs.neuralinsights.ai
Model Version: 1.0
Last Updated: November 2025
═══════════════════════════════════════════════════════════
"""
    
    return send_file(
        io.BytesIO(template_content.encode()),
        mimetype='text/plain',
        as_attachment=True,
        download_name='delivery_risk_training_template.txt'
    )

@app.route('/download/client')
def download_client():
    import os
    
    # Check if client.py exists in the same directory
    client_path = os.path.join(os.path.dirname(__file__), 'client.py')
    
    if os.path.exists(client_path):
        return send_file(
            client_path,
            mimetype='text/x-python',
            as_attachment=True,
            download_name='client.py'
        )
    else:
        # Return an error message if file doesn't exist
        return jsonify({
            'error': 'client.py not found in the current directory',
            'message': 'Please ensure client.py is in the same directory as this Flask application'
        }), 404

@app.route('/predict', methods=['POST'])
def predict():
    import torch
    import os
    
    data = request.json
    
    # Check if model file exists
    MODEL_EXPORT_FILE = "delivery_risk_model.pkl"
    if not os.path.exists(MODEL_EXPORT_FILE):
        return jsonify({
            'error': 'Model file not found',
            'message': f'Please ensure {MODEL_EXPORT_FILE} is in the same directory as this Flask application'
        }), 404
    
    try:
        # Load the saved model and preprocessor
        with open(MODEL_EXPORT_FILE, 'rb') as f:
            package = pickle.load(f)
        
        model_state = package['model_state_dict']
        preprocessor = package['preprocessor']
        input_size = package['input_size']
        
        # Define model architecture
        class SimpleRegressionANN(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layer_1 = torch.nn.Linear(input_size, 64)
                self.relu_1 = torch.nn.ReLU()
                self.layer_2 = torch.nn.Linear(64, 32)
                self.relu_2 = torch.nn.ReLU()
                self.layer_3 = torch.nn.Linear(32, 16)
                self.relu_3 = torch.nn.ReLU()
                self.layer_out = torch.nn.Linear(16, 1)
            
            def forward(self, x):
                x = self.relu_1(self.layer_1(x))
                x = self.relu_2(self.layer_2(x))
                x = self.relu_3(self.layer_3(x))
                x = self.layer_out(x)
                return x
        
        # Reconstruct model
        model = SimpleRegressionANN(input_size)
        model.load_state_dict(model_state)
        model.eval()
        
        # Prepare input data from user
        example = pd.DataFrame([{
            'late_delivery_risk': float(data['late_delivery_risk']),
            'order_item_profit_ratio': float(data['order_item_profit_ratio']),
            'order_item_quantity': float(data['order_item_quantity']),
            'order_year': int(data['order_year']),
            'order_month': int(data['order_month']),
            'order_day': int(data['order_day']),
            'order_hour': int(data['order_hour']),
            'order_minute': int(data['order_minute']),
            'order_weekend': int(data['order_weekend']),
            'ship_year': int(data['ship_year']),
            'ship_month': int(data['ship_month']),
            'ship_day': int(data['ship_day']),
            'ship_hour': int(data['ship_hour']),
            'ship_minute': int(data['ship_minute']),
            'ship_weekend': int(data['ship_weekend']),
            'delivery_status': data['delivery_status'],
            'customer_segment': data['customer_segment'],
            'department_name': data['department_name'],
            'order_status': data['order_status'],
            'shipping_mode': data['shipping_mode']
        }])
        
        # Preprocess and predict
        X_processed = preprocessor.transform(example).astype(np.float32)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(X_tensor).numpy().flatten()[0]
        
        return jsonify({'prediction': f'{prediction:.3f}'})
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

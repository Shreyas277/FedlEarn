#!/bin/bash
# A script to run the federated learning system using wait for synchronization

# --- SETUP AND CLEANUP ---
echo "Stopping any existing server and clients..."
pkill -f "python server.py"
pkill -f "python client.py"

echo "Installing dependencies..."
pip install -q flwr pandas scikit-learn numpy catboost

# --- 1. START SERVER ---
echo "--- 1. Starting server in background ---"
python server.py > server.log 2>&1 &
SERVER_PID=$!
echo "Server running with PID $SERVER_PID. Log: server.log"
sleep 5 # Give server process time to fully initialize Flower server

# --- 2. START CLIENT BATCHES (Rounds) ---

# We only run two batches based on your existing code structure.
# NOTE: Your server runs 40 total rounds. This script only runs 2 rounds.

echo "--- 2. Starting clients for Round 1 ---"
python client.py --csv_file client11.csv > client1_ann.log 2>&1 &
CLIENT1_PID=$!
python client.py --csv_file client21.csv > client2_ann.log 2>&1 &
CLIENT2_PID=$!

echo "Waiting for clients 1 & 2 to complete Round 1 aggregation..."
wait $CLIENT1_PID $CLIENT2_PID
echo "Clients 1 & 2 disconnected. Round 1 complete."

# Ensure a brief pause before injecting the next batch, allowing the server to save the model.
sleep 2

echo "--- 3. Starting clients for Round 2 ---"
python client.py --csv_file client31.csv > client3_ann.log 2>&1 &
CLIENT3_PID=$!
python client.py --csv_file client41.csv > client4_ann.log 2>&1 &
CLIENT4_PID=$!

echo "Waiting for clients 3 & 4 to complete Round 2 aggregation..."
wait $CLIENT3_PID $CLIENT4_PID
echo "Clients 3 & 4 disconnected. Round 2 complete."
sleep 2


echo "--- 2. Starting clients for Round 3 ---"
python client.py --csv_file client51.csv > client5_ann.log 2>&1 &
CLIENT5_PID=$!
python client.py --csv_file client61.csv > client6_ann.log 2>&1 &
CLIENT6_PID=$!

echo "Waiting for clients 5 & 6 to complete Round 3 aggregation..."
wait $CLIENT5_PID $CLIENT6_PID
echo "Clients 5 & 6 disconnected. Round 3 complete."

# Ensure a brief pause before injecting the next batch, allowing the server to save the model.
sleep 2

# If you wanted to run all 40 rounds automatically, you would modify the server 
# to exit after 40 rounds (which it does) and then kill the server here.

# Kill the server
echo "Stopping server process (PID $SERVER_PID)..."
kill $SERVER_PID

echo "You can check 'server.log' for results."

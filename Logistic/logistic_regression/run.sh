#!/bin/bash
# A script to run the federated learning system

# Stop any old processes
echo "Stopping any existing server and clients..."
pkill -f "python server.py"
pkill -f "python client.py"

# Install dependencies
echo "Installing dependencies..."
pip install -q flwr pandas scikit-learn numpy

# 1. Initialize the model (RUN THIS FIRST, ONLY ONCE)
# This creates the 'final_model.npz' file that the server needs
echo "--- 1. Initializing the model ---"
python init_model.py
if [ $? -ne 0 ]; then
    echo "Model initialization failed! Exiting."
    exit 1
fi
echo "Model initialized."

# 2. Start the server in the background
echo "--- 2. Starting server in background ---"
python server.py > server_withp.log 2>&1 &
SERVER_PID=$!
echo "Server running with PID $SERVER_PID. Log: server.log"
sleep 5 # Give server time to start

# 3. Start 6 clients, each with their own data
# We'll start 3, wait, then start 3 more.
# The server is set to min_available_clients=2, so it will run in batches.
echo "--- 3. Starting clients ---"

echo "Starting client1.csv and client2.csv (First batch)"
python client.py --csv_file client1.csv > client1_withp.log 2>&1 &
python client.py --csv_file client2.csv > client2_withp.log 2>&1 &
sleep 10 # Wait for them to connect and finish round 1

echo "Starting client3.csv and client4.csv (Second batch)"
python client.py --csv_file client3.csv > client3_withp.log 2>&1 &
python client.py --csv_file client4.csv > client4_withp.log 2>&1 &
sleep 10 # Wait for them to connect and finish round 2

echo "Starting client5.csv and client6.csv (Third batch)"
python client.py --csv_file client5.csv > client5_withp.log 2>&1 &
python client.py --csv_file client6.csv > client6_withp.log 2>&1 &
sleep 10 # Wait for them to connect and finish round 3

echo "--- All clients have run ---"
echo "You can check 'server.log' to see the aggregation results."
echo "The final aggregated model is saved in 'final_model.npz'."
echo "To stop the server, run: kill $SERVER_PID"


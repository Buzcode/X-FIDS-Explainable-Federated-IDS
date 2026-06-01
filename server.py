import flwr as fl

# Define modern strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,      # Start training as soon as 2 clients connect
    min_available_clients=2 # Minimum clients needed
)

# Start server using the modern stable API
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
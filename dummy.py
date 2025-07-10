from zenml.client import Client

tracker = Client().active_stack.experiment_tracker
print("🧭 Experiment Tracker Name:", tracker.name)
print("📍 Tracking URI:", tracker.get_tracking_uri())

import os
import json
import datetime

class StateManager:
    def __init__(self, state_file="database/tracker_state.json", tracker=None):
        self.state_file = state_file
        self.available_balance = None
        self.last_render_time = None
        self.saved_data_path = None
        self.tracker = tracker  # Store reference to tracker for save_state
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as file:
                try:
                    state = json.load(file)
                    self.last_render_time = datetime.datetime.fromisoformat(state.get("last_render_time"))
                    self.saved_data_path = state.get("saved_data_path")
                    self.available_balance = state.get("available_balance")
                    print(self.available_balance)

                except Exception as e:
                    print(f"Error loading state: {e}")

    def save_state(self):
        if self.tracker is None:
            print("Warning: No tracker reference in StateManager, cannot save available_fund")
            return

        state = {
            "last_render_time": self.last_render_time.isoformat() if self.last_render_time else None,
            "saved_data_path": self.saved_data_path,
            "available_balance": self.tracker.available_fund
        }
        with open(self.state_file, "w") as file:
            json.dump(state, file)

    def can_run_tracker(self):
        if self.last_render_time is None:
            return True  # First-time run
        return self.last_render_time.date() < datetime.datetime.now().date()

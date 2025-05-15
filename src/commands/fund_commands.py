from .command_base import Command

class AddFundCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        try:
            amount_str = input("Enter amount to add to available fund: ").strip()
            amount = float(amount_str)
            if amount <= 0:
                print("Please enter a positive amount.")
                return
            self.tracker.available_fund += amount
            self.tracker.state_manager.save_state()
            print(f"Available fund increased. New balance: {self.tracker.available_fund}")
        except Exception as e:
            print(f"Error adding fund: {e}")

class ExtractFundCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        try:
            amount_str = input("Enter amount to extract from available fund: ").strip()
            amount = float(amount_str)
            if amount <= 0:
                print("Please enter a positive amount.")
                return
            if amount > self.tracker.available_fund:
                print(f"Insufficient funds. Available fund: {self.tracker.available_fund}")
                return
            self.tracker.available_fund -= amount
            self.tracker.state_manager.save_state()
            print(f"Available fund decreased. New balance: {self.tracker.available_fund}")
        except Exception as e:
            print(f"Error extracting fund: {e}")

from src.commands.command_base import Command
from src.utils.util_manager import UtilManager

class UtilCommand(Command):
    def __init__(self, tracker):
        self.util_manager = UtilManager(tracker)

    def execute(self):
        self.util_manager.execute_util_command()

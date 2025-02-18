import logging
import os
from rich.logging import RichHandler

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logs_dir = os.path.join(root_dir, "logs")

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)

shell_handler = RichHandler()
file_handler = logging.FileHandler(os.path.join(logs_dir, "debug.log"))

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

fmt_shell = "%(message)s"
fmt_file = (
    "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
)

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)

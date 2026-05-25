import logging
import logging.handlers
from rich.logging import RichHandler
import os
import re

EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))

def clean_rich_markup(text):
    return re.sub(r'\[\/?[a-zA-Z0-9\s#_,]+\]', '', text)

class PlainFileFormatter(logging.Formatter):
    def format(self, record):
        original_msg = record.msg
        if isinstance(record.msg, str):
            record.msg = clean_rich_markup(record.msg).replace('\\', '')
        
        result = super().format(record)
        record.msg = original_msg
        return result

class Log(logging.Logger):
    
    def info_color(self, msg: str, *args, **kwargs):
        super().info(f"[i blue bold]{msg}",extra = {"markup" : True}, stacklevel=2)
    
    def warning_color(self, msg: str, *args, **kwargs):
        super().warning(f"[i yellow bold]{msg}", extra= {"markup" : True}, stacklevel=2)
    
    def exception_color(self, msg: str, *args, **kwargs):
        super().exception(f"[bold red] \[!] {msg}", extra= {"markup" : True}, stacklevel=2)
    def debug_color(self, msg: str, *args, **kwargs): 
            super().debug(f"[i bold green]{msg}", extra= {"markup" : True}, stacklevel=2)

logging.setLoggerClass(Log)

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
file_handler = logging.FileHandler(f"{EXECUTION_PATH}/logs.txt", mode = "w", encoding= "utf-8")
file_formatter = PlainFileFormatter(FORMAT, datefmt="[%X]")
file_handler.setFormatter(file_formatter)
console_handler = RichHandler(markup=True, show_path=False)


logging.basicConfig(
    level = "NOTSET", format=FORMAT, datefmt="[%X]", handlers=[console_handler, file_handler]
)

logger = logging.getLogger("llm-eval")
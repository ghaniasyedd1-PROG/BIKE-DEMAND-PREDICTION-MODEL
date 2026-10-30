"""Custom exception that includes the source file and line number."""
import sys

class CustomException(Exception):
    def __init__(self, message: str):
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is not None:
            frame = exc_tb.tb_frame
            filename = frame.f_code.co_filename
            line = exc_tb.tb_lineno
            detail = f"{message} | File: {filename} | Line: {line}"
        else:
            detail = str(message)
        super().__init__(detail)

import sys
import pathlib
from string import Template
import logging
import inspect

from loguru import logger


class PromptUtils:

    def __init__(self, prompt_path: pathlib.Path):
        """
        Prompt configuration class

        Args:
            prompt_path (pathlib.Path): Path to prompt file
        """
        self.prompt_path: pathlib.Path = prompt_path
        self.sys_prompt: str = self.get_sys_prompt()
        self.original_prompt: str = self.get_original_prompt()

    def get_sys_prompt(self):
        """
        System prompt, the first line of the md file

        Args:
            prompt_path (pathlib.Path): Path to prompt file
        """
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.readline().strip()

    def get_original_prompt(self):
        """
        Get user prompt, content from the second line to the end of md file

        Args:
            prompt_path (pathlib.Path): Path to prompt file
        """
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[1:]).strip()

    def get_prompt(self, **kwargs):
        """
        Get prompt, content from the second line to the end of md file

        Args:
            prompt_path (pathlib.Path): Path to prompt file
            **kwargs: Keyword arguments
        """
        return Template(self.original_prompt).substitute(**kwargs)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logger.remove()
logger.add(sink=sys.stderr, level="INFO")

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

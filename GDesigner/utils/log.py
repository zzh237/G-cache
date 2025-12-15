"""Dual logger for G-cache project - outputs to both console and file"""
import logging
import sys
from datetime import datetime
from pathlib import Path


class DualLogger:
    """Logger that writes to both console and file"""
    
    def __init__(self, log_dir: str = None, task: str = "gsm8k", prefix: str = "cache_API"):
        # Get project root (G-cache directory)
        project_root = Path(__file__).parent.parent.parent
        
        # Create log directory: G-cache/log/{task}/
        if log_dir is None:
            log_dir = project_root / "log" / task
        else:
            log_dir = Path(log_dir) / task
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_filename = f"{prefix}_{task}_{timestamp}.log"
        self.log_path = log_dir / log_filename
        
        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        print(f"📝 Logging to: {self.log_path}")
    
    def write(self, message):
        """Write to both console and file"""
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """Flush both outputs"""
        self.original_stdout.flush()
        self.log_file.flush()
    
    def close(self):
        """Close log file and restore original stdout"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def __enter__(self):
        """Context manager entry"""
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


def setup_logger(task: str = "gsm8k", prefix: str = "cache_API", log_dir: str = None):
    """
    Setup dual logger for the entire session
    
    Args:
        task: Task name (e.g., "gsm8k", "aime2024")
        prefix: Log file prefix (e.g., "cache_API", "cache_LOCAL")
        log_dir: Custom log directory (optional)
    
    Returns:
        DualLogger instance
    
    Example:
        >>> from GDesigner.utils.log import setup_logger
        >>> logger = setup_logger(task="gsm8k", prefix="cache_API")
        >>> print("This goes to both console and log file")
        >>> logger.close()  # When done
    """
    return DualLogger(log_dir=log_dir, task=task, prefix=prefix)


# Keep original logger for backward compatibility
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

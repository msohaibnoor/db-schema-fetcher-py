"""
Logging configuration for MySQL Schema Fetcher
"""
import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import config

# Initialize rich console
console = Console()

class SchemaLogger:
    """Custom logger for schema extraction operations"""
    
    def __init__(self):
        self.setup_logger()
        self.progress = None
    
    def setup_logger(self):
        """Configure loguru logger"""
        # Remove default logger
        logger.remove()
        
        # Add console logger with rich formatting
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=config.processing.log_level,
            colorize=True
        )
        
        # Add file logger
        log_file = config.processing.output_dir / "schema_extraction.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days"
        )
    
    def start_progress(self, description: str):
        """Start progress bar"""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )
        self.progress.start()
        return self.progress.add_task(description, total=100)
    
    def update_progress(self, task_id, advance: int = 1, description: str = None):
        """Update progress bar"""
        if self.progress:
            self.progress.update(task_id, advance=advance, description=description)
    
    def stop_progress(self):
        """Stop progress bar"""
        if self.progress:
            self.progress.stop()
            self.progress = None
    
    def log_section(self, title: str):
        """Log a section header"""
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue] {title.upper()} [/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")
    
    def log_success(self, message: str):
        """Log success message"""
        console.print(f"[bold green]✓[/bold green] {message}")
        logger.success(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        console.print(f"[bold red]✗[/bold red] {message}")
        logger.error(message)
    
    def log_info(self, message: str):
        """Log info message"""
        console.print(f"[bold cyan]ℹ[/bold cyan] {message}")
        logger.info(message)
    
    def log_table_info(self, table_name: str, column_count: int, row_count: int = None):
        """Log table information"""
        msg = f"Table: {table_name} ({column_count} columns"
        if row_count is not None:
            msg += f", ~{row_count:,} rows"
        msg += ")"
        
        console.print(f"  [dim cyan]{msg}[/dim cyan]")
        logger.debug(msg)

# Global logger instance
schema_logger = SchemaLogger()
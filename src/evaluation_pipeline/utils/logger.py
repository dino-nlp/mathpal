"""
Enhanced logging utilities for the evaluation pipeline.
"""

import logging
import sys
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint


class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for logs."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with rich formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create rich handler
    rich_handler = RichHandler(
        console=Console(),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    
    # Create formatter
    formatter = logging.Formatter(
        "%(name)s - %(message)s",
        datefmt="[%X]"
    )
    rich_handler.setFormatter(formatter)
    
    logger.addHandler(rich_handler)
    return logger


def print_evaluation_header():
    """Print a beautiful evaluation header."""
    console = Console()
    
    header = Panel(
        Text("🧮 MathPal Evaluation Pipeline", style="bold blue"),
        subtitle="[dim]Comprehensive AI Model Assessment[/dim]",
        border_style="blue"
    )
    console.print(header)


def print_evaluation_results(results: dict, model_name: str = "Unknown Model"):
    """Print evaluation results in a beautiful format."""
    console = Console()
    
    # Create results table
    table = Table(title=f"📊 Evaluation Results for {model_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", justify="center")
    table.add_column("Status", style="green", justify="center")
    table.add_column("Description", style="white")
    
    # Define score ranges and descriptions for different metric types
    score_ranges = {
        # For 0-1 scale metrics (Opik, custom metrics)
        (0.0, 0.3): ("🔴 Poor", "Needs significant improvement"),
        (0.3, 0.5): ("🟡 Fair", "Below average performance"),
        (0.5, 0.7): ("🟠 Average", "Acceptable but room for improvement"),
        (0.7, 0.85): ("🟢 Good", "Above average performance"),
        (0.85, 1.0): ("🟢 Excellent", "Outstanding performance")
    }
    
    # For 1-10 scale metrics (LLM-as-a-judge)
    llm_score_ranges = {
        (1.0, 3.0): ("🔴 Poor", "Needs significant improvement"),
        (3.0, 5.0): ("🟡 Fair", "Below average performance"),
        (5.0, 7.0): ("🟠 Average", "Acceptable but room for improvement"),
        (7.0, 8.5): ("🟢 Good", "Above average performance"),
        (8.5, 10.0): ("🟢 Excellent", "Outstanding performance")
    }
    
    def get_score_status(score: float, metric_name: str = "") -> tuple:
        """Get status and description for a score."""
        # Use LLM scale for LLM-as-a-judge metrics
        if any(llm_metric in metric_name.lower() for llm_metric in ['accuracy', 'completeness', 'clarity', 'relevance', 'helpfulness']):
            ranges = llm_score_ranges
        else:
            ranges = score_ranges
            
        for (min_score, max_score), (status, description) in ranges.items():
            if min_score <= score < max_score:
                return status, description
        return "🔴 Poor", "Invalid score"
    
    # Add rows to table
    for metric, score in results.items():
        if metric == "overall_score":
            continue
            
        if isinstance(score, (int, float)):
            status, description = get_score_status(score, metric)
            table.add_row(
                metric.replace("_", " ").title(),
                f"{score:.3f}",
                status,
                description
            )
    
    # Add overall score row
    if "overall_score" in results:
        overall_score = results["overall_score"]
        status, description = get_score_status(overall_score, "overall")
        table.add_row(
            "[bold]Overall Score[/bold]",
            f"[bold]{overall_score:.3f}[/bold]",
            f"[bold]{status}[/bold]",
            f"[bold]{description}[/bold]"
        )
    
    console.print(table)
    
    # Print detailed analysis
    console.print("\n")
    analysis_table = Table(title="📈 Detailed Analysis")
    analysis_table.add_column("Category", style="cyan")
    analysis_table.add_column("Best Metric", style="green")
    analysis_table.add_column("Worst Metric", style="red")
    analysis_table.add_column("Recommendation", style="yellow")
    
    # Analyze by category
    categories = {
        "Core Metrics": ["hallucination", "context_precision", "context_recall", "answer_relevance", "usefulness"],
        "Vietnamese Math": ["mathematical_accuracy", "vietnamese_language_quality"],
        "LLM-as-a-Judge": ["accuracy", "completeness", "clarity", "relevance", "helpfulness"]
    }
    
    for category, metrics in categories.items():
        category_scores = {k: v for k, v in results.items() if k in metrics and isinstance(v, (int, float))}
        
        if category_scores:
            best_metric = max(category_scores.items(), key=lambda x: x[1])
            worst_metric = min(category_scores.items(), key=lambda x: x[1])
            
            # Generate recommendation
            if best_metric[1] >= 0.8:
                recommendation = "Excellent performance in this category"
            elif worst_metric[1] <= 0.3:
                recommendation = "Focus on improving weakest areas"
            else:
                recommendation = "Balanced performance, room for improvement"
            
            analysis_table.add_row(
                category,
                f"{best_metric[0].replace('_', ' ').title()} ({best_metric[1]:.3f})",
                f"{worst_metric[0].replace('_', ' ').title()} ({worst_metric[1]:.3f})",
                recommendation
            )
    
    console.print(analysis_table)
    
    # Print summary
    if "overall_score" in results:
        overall_score = results["overall_score"]
        if overall_score >= 0.85:
            summary = Panel(
                Text("🎉 Excellent Performance!", style="bold green"),
                subtitle="[dim]The model shows outstanding capabilities across all metrics[/dim]",
                border_style="green"
            )
        elif overall_score >= 0.7:
            summary = Panel(
                Text("✅ Good Performance", style="bold blue"),
                subtitle="[dim]The model performs above average with some areas for improvement[/dim]",
                border_style="blue"
            )
        elif overall_score >= 0.5:
            summary = Panel(
                Text("⚠️ Average Performance", style="bold yellow"),
                subtitle="[dim]The model needs improvement in several key areas[/dim]",
                border_style="yellow"
            )
        else:
            summary = Panel(
                Text("❌ Poor Performance", style="bold red"),
                subtitle="[dim]The model requires significant improvement across all metrics[/dim]",
                border_style="red"
            )
        
        console.print(summary)


def print_model_info(model_name: str, model_path: str, device: str):
    """Print model information."""
    console = Console()
    
    info_table = Table(title="🤖 Model Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Model Name", model_name)
    info_table.add_row("Model Path", model_path)
    info_table.add_row("Device", device)
    
    console.print(info_table)


def print_dataset_info(dataset_name: str, sample_count: int, source: str):
    """Print dataset information."""
    console = Console()
    
    info_table = Table(title="📚 Dataset Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Dataset Name", dataset_name)
    info_table.add_row("Sample Count", str(sample_count))
    info_table.add_row("Source", source)
    
    console.print(info_table)


def print_progress_bar(description: str, total: int):
    """Create and return a progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console()
    )


def print_step_header(step_name: str, step_number: int, total_steps: int):
    """Print a step header."""
    console = Console()
    
    header = Panel(
        Text(f"Step {step_number}/{total_steps}: {step_name}", style="bold blue"),
        border_style="blue"
    )
    console.print(header)


def print_error_summary(error: str, context: str = ""):
    """Print error summary."""
    console = Console()
    
    error_panel = Panel(
        Text(f"❌ Error: {error}", style="bold red"),
        subtitle=f"[dim]{context}[/dim]" if context else None,
        border_style="red"
    )
    console.print(error_panel)


def print_success_message(message: str):
    """Print success message."""
    console = Console()
    
    success_panel = Panel(
        Text(f"✅ {message}", style="bold green"),
        border_style="green"
    )
    console.print(success_panel)


def print_warning_message(message: str):
    """Print warning message."""
    console = Console()
    
    warning_panel = Panel(
        Text(f"⚠️ {message}", style="bold yellow"),
        border_style="yellow"
    )
    console.print(warning_panel)

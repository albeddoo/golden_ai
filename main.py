#!/usr/bin/env python3
"""
GOLD AI - Live Multi-Agent Gold Market Intelligence & Forecasting Terminal
Main Application Entry Point

This application provides real-time analysis of XAU/USD using a multi-agent system
that collects evidence from multiple independent sources, detects market regimes,
calibrates probabilities, and generates transparent final decisions.

IMPORTANT: This is for research, education, backtesting, and paper/simulation only.
No automatic real-money trading or broker order execution is implemented.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gold_ai.app.orchestrator import Orchestrator
from gold_ai.core.config import load_config
from gold_ai.core.logging_setup import setup_logging


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GOLD AI - Live Multi-Agent Gold Market Intelligence Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default configuration
  python main.py --config my_config.yaml  # Run with custom configuration
  python main.py --mode backtest          # Run in backtesting mode
  python main.py --headless               # Run without GUI (logging only)
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'backtest', 'paper'],
        default='paper',
        help='Application mode: live (real data), backtest (historical), paper (simulation)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI (logging and analysis only)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory for data storage (default: data)'
    )
    
    return parser.parse_args()


def check_dependencies():
    """Verify all required dependencies are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(
        level=args.log_level,
        log_dir=Path(args.data_dir) / 'logs'
    )
    
    logger = logging.getLogger('gold_ai.main')
    logger.info("=" * 60)
    logger.info("GOLD AI - Live Multi-Agent Gold Market Intelligence Terminal")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Headless: {args.headless}")
    
    # Check dependencies
    check_dependencies()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize orchestrator
    try:
        orchestrator = Orchestrator(
            config=config,
            mode=args.mode,
            headless=args.headless,
            data_dir=Path(args.data_dir)
        )
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        sys.exit(1)
    
    # Start application
    try:
        logger.info("Starting GOLD AI Terminal...")
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("User interrupted application")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down GOLD AI Terminal...")
        orchestrator.shutdown()
        logger.info("Application terminated cleanly")


if __name__ == '__main__':
    main()

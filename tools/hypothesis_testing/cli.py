#!/usr/bin/env python3
"""
CLI for testing 7D BVP hypotheses.

This tool provides modular command-based testing of hypotheses
for the 7D BVP theory framework.
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from .commands import TestStep0Command, TestStep1Command, TestAllCommand, TestStep2Command


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test 7D BVP hypotheses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  test-step-0    Test Step 0: 7D BVP Structure Validation
  test-step-1    Test Step 1: Power Law Analyzer - Stepwise Structure  
  test-step-2    Test Step 2: 7D FFT Solver
  test-all       Run all available tests

Examples:
  python cli.py test-step-0 --verbose
  python cli.py test-step-1
  python cli.py test-all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test Step 0 command
    step0_parser = subparsers.add_parser('test-step-0', help='Test Step 0: 7D BVP Structure')
    step0_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Test Step 1 command
    step1_parser = subparsers.add_parser('test-step-1', help='Test Step 1: Power Law Analyzer')
    step1_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Test Step 2 command
    step2_parser = subparsers.add_parser('test-step-2', help='Test Step 2: 7D FFT Solver')
    step2_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    # Test All command
    all_parser = subparsers.add_parser('test-all', help='Run all available tests')
    all_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    try:
        if args.command == 'test-step-0':
            command = TestStep0Command(verbose=args.verbose)
            result = command.execute()
            command.print_result(result)
            sys.exit(0 if result.get("success", False) else 1)
            
        elif args.command == 'test-step-1':
            command = TestStep1Command(verbose=args.verbose)
            result = command.execute()
            command.print_result(result)
            sys.exit(0 if result.get("success", False) else 1)
            
        elif args.command == 'test-all':
            command = TestAllCommand(verbose=args.verbose)
            result = command.execute()
            sys.exit(0 if result.get("success", False) else 1)
        
        elif args.command == 'test-step-2':
            command = TestStep2Command(verbose=args.verbose)
            result = command.execute()
            command.print_result(result)
            sys.exit(0 if result.get("success", False) else 1)
            
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
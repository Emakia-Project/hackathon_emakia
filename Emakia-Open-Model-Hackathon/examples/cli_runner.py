#!/usr/bin/env python3
"""
CLI Runner for the Emakia Validator Agent.

This module provides a command-line interface for the validation agent.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.main import EmakiaValidatorAgent


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Emakia Validator Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single piece of content
  python cli_runner.py --content "Hello world" --type text

  # Validate content from file
  python cli_runner.py --file content.txt --type text

  # Batch validation
  python cli_runner.py --batch batch_content.txt --type text

  # Health check
  python cli_runner.py --health

  # Interactive mode
  python cli_runner.py --interactive
        """
    )
    
    # Content input options
    content_group = parser.add_mutually_exclusive_group()
    content_group.add_argument(
        "--content",
        help="Content to validate"
    )
    content_group.add_argument(
        "--file",
        help="File containing content to validate"
    )
    content_group.add_argument(
        "--batch",
        help="File containing multiple content items (one per line)"
    )
    
    # Other options
    parser.add_argument(
        "--type",
        default="text",
        choices=["text", "image", "video"],
        help="Content type (default: text)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Perform health check"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    try:
        agent = EmakiaValidatorAgent(args.config)
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        sys.exit(1)
    
    # Health check
    if args.health:
        health = agent.health_check()
        print(json.dumps(health, indent=2))
        return
    
    # Interactive mode
    if args.interactive:
        await interactive_mode(agent)
        return
    
    # Process content
    if args.content:
        await process_single_content(agent, args.content, args.type, args.output, args.verbose)
    elif args.file:
        await process_file_content(agent, args.file, args.type, args.output, args.verbose)
    elif args.batch:
        await process_batch_content(agent, args.batch, args.type, args.output, args.verbose)
    else:
        print("No content specified. Use --content, --file, --batch, --health, or --interactive")
        parser.print_help()


async def process_single_content(agent, content, content_type, output_file, verbose):
    """Process single content item."""
    print(f"Validating content (type: {content_type})...")
    
    try:
        result = await agent.validate_content(content, content_type)
        
        if verbose:
            print(json.dumps(result, indent=2))
        else:
            print_summary(result)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        sys.exit(1)


async def process_file_content(agent, file_path, content_type, output_file, verbose):
    """Process content from file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        await process_single_content(agent, content, content_type, output_file, verbose)
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)


async def process_batch_content(agent, file_path, content_type, output_file, verbose):
    """Process batch content from file."""
    try:
        with open(file_path, 'r') as f:
            contents = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(contents)} items...")
        
        results = await agent.batch_validate(contents, content_type)
        
        if verbose:
            print(json.dumps(results, indent=2))
        else:
            print_batch_summary(results)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_file}")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        sys.exit(1)


async def interactive_mode(agent):
    """Run in interactive mode."""
    print("Emakia Validator Agent - Interactive Mode")
    print("Enter content to validate (or 'quit' to exit)")
    print("Commands:")
    print("  quit - Exit the application")
    print("  health - Perform health check")
    print("  metrics - Show current metrics")
    print("  help - Show this help")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'health':
                health = agent.health_check()
                print(json.dumps(health, indent=2))
            elif user_input.lower() == 'metrics':
                metrics = agent.get_metrics()
                print(json.dumps(metrics, indent=2))
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  quit - Exit the application")
                print("  health - Perform health check")
                print("  metrics - Show current metrics")
                print("  help - Show this help")
            elif user_input:
                print("Validating content...")
                result = await agent.validate_content(user_input, "text")
                print_summary(result)
            else:
                continue
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def print_summary(result):
    """Print a summary of validation results."""
    status = result.get('status', 'unknown')
    validation = result.get('validation', {})
    classification = result.get('classification', {})
    
    print(f"\nStatus: {status}")
    print(f"Valid: {validation.get('is_valid', False)}")
    print(f"Category: {classification.get('category', 'unknown')}")
    print(f"Confidence: {classification.get('confidence', 0.0):.2f}")
    
    violations = validation.get('violations', [])
    if violations:
        print("Violations:")
        for violation in violations:
            print(f"  - {violation}")
    
    suggestions = validation.get('suggestions', [])
    if suggestions:
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")


def print_batch_summary(results):
    """Print a summary of batch validation results."""
    total = len(results)
    valid_count = sum(1 for r in results if r.get('validation', {}).get('is_valid', False))
    success_rate = valid_count / total if total > 0 else 0.0
    
    print(f"\nBatch Processing Summary:")
    print(f"Total items: {total}")
    print(f"Valid items: {valid_count}")
    print(f"Invalid items: {total - valid_count}")
    print(f"Success rate: {success_rate:.1%}")
    
    # Category distribution
    categories = {}
    for result in results:
        category = result.get('classification', {}).get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print("Category distribution:")
    for category, count in categories.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    asyncio.run(main())

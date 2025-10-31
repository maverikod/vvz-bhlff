"""
Test All: Run all available tests.
"""

from typing import Dict, Any, List
from .base import BaseCommand
from .test_step_0 import TestStep0Command
from .test_step_1 import TestStep1Command


class TestAllCommand(BaseCommand):
    """Test All: Run all available tests."""

    def execute(self) -> Dict[str, Any]:
        """Execute all tests."""
        self.logger.info("Running all available tests...")

        # Run all tests
        test_commands = [
            TestStep0Command(verbose=self.verbose),
            TestStep1Command(verbose=self.verbose),
        ]

        results = []
        for command in test_commands:
            result = command.execute()
            results.append(result)
            command.print_result(result)
            print()  # Empty line between tests

        # Calculate summary
        passed = sum(1 for r in results if r.get("success", False))
        total = len(results)

        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print("=" * 60)

        return {
            "name": "All Tests",
            "success": passed == total,
            "details": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "results": results,
            },
        }

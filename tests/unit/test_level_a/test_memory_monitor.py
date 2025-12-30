"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Memory monitoring utility for tests.

Physical Meaning:
    Provides memory monitoring capabilities for tests to ensure
    that memory usage stays within acceptable limits and to detect
    memory leaks or excessive memory consumption.

Example:
    >>> with MemoryMonitor(max_memory_mb=1000) as monitor:
    ...     # Run test code
    ...     current_memory = monitor.get_current_memory_mb()
    ...     peak_memory = monitor.get_peak_memory_mb()
"""

import os
import sys
import time
import threading
import psutil
from typing import Optional, Dict, Any
from contextlib import contextmanager


class MemoryMonitor:
    """
    Memory monitor for test processes.
    
    Physical Meaning:
        Monitors memory usage of the current process and optionally
        kills the process if memory limits are exceeded.
        
    Attributes:
        max_memory_mb (float): Maximum allowed memory in MB.
        check_interval (float): Interval between memory checks in seconds.
        process (psutil.Process): Process being monitored.
        peak_memory_mb (float): Peak memory usage in MB.
        start_memory_mb (float): Initial memory usage in MB.
        _monitoring (bool): Whether monitoring is active.
        _monitor_thread (threading.Thread): Thread running the monitor.
    """
    
    def __init__(
        self,
        max_memory_mb: float = 1000.0,
        check_interval: float = 0.1,
        kill_on_exceed: bool = False,
    ):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_mb (float): Maximum allowed memory in MB.
            check_interval (float): Interval between memory checks in seconds.
            kill_on_exceed (bool): Whether to kill process if limit exceeded.
        """
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.kill_on_exceed = kill_on_exceed
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0.0
        self.start_memory_mb = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_samples: list = []
    
    def start(self) -> None:
        """Start memory monitoring."""
        if self._monitoring:
            return
        
        self.start_memory_mb = self._get_memory_mb()
        self.peak_memory_mb = self.start_memory_mb
        self._monitoring = True
        self._memory_samples = []
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in separate thread)."""
        while self._monitoring:
            try:
                current_memory = self._get_memory_mb()
                self._memory_samples.append((time.time(), current_memory))
                
                # Update peak memory
                if current_memory > self.peak_memory_mb:
                    self.peak_memory_mb = current_memory
                
                # Check if limit exceeded
                if current_memory > self.max_memory_mb:
                    error_msg = (
                        f"Memory limit exceeded: {current_memory:.2f} MB > "
                        f"{self.max_memory_mb:.2f} MB"
                    )
                    if self.kill_on_exceed:
                        print(f"ERROR: {error_msg}", file=sys.stderr)
                        self.process.kill()
                    else:
                        print(f"WARNING: {error_msg}", file=sys.stderr)
                
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in memory monitor: {e}", file=sys.stderr)
                break
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._get_memory_mb()
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory_mb
    
    def get_memory_increase_mb(self) -> float:
        """Get memory increase from start in MB."""
        return self.peak_memory_mb - self.start_memory_mb
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "start_memory_mb": self.start_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.get_current_memory_mb(),
            "memory_increase_mb": self.get_memory_increase_mb(),
            "max_memory_mb": self.max_memory_mb,
            "samples_count": len(self._memory_samples),
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


@contextmanager
def memory_monitor_context(
    max_memory_mb: float = 1000.0,
    check_interval: float = 0.1,
    kill_on_exceed: bool = False,
):
    """
    Context manager for memory monitoring.
    
    Physical Meaning:
        Provides a convenient context manager for memory monitoring
        in tests, automatically starting and stopping the monitor.
        
    Args:
        max_memory_mb (float): Maximum allowed memory in MB.
        check_interval (float): Interval between memory checks in seconds.
        kill_on_exceed (bool): Whether to kill process if limit exceeded.
        
    Yields:
        MemoryMonitor: Memory monitor instance.
        
    Example:
        >>> with memory_monitor_context(max_memory_mb=500) as monitor:
        ...     # Run test code
        ...     peak = monitor.get_peak_memory_mb()
    """
    monitor = MemoryMonitor(
        max_memory_mb=max_memory_mb,
        check_interval=check_interval,
        kill_on_exceed=kill_on_exceed,
    )
    try:
        monitor.start()
        yield monitor
    finally:
        monitor.stop()


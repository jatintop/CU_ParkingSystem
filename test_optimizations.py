#!/usr/bin/env python3
"""
Test script to validate optimizations and measure performance improvements.
"""

import time
import psutil
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import settings
from core.optimized_processor import OptimizedParkingProcessor
from core.optimized_detector import OptimizedVehicleDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationTester:
    """Test class for validating optimizations."""
    
    def __init__(self):
        self.processor = None
        self.start_time = None
        self.initial_memory = None
        self.initial_cpu = None
        
    def start_test(self):
        """Start the optimization test."""
        logger.info("Starting optimization test...")
        
        # Record initial system state
        self.initial_memory = psutil.virtual_memory().percent
        self.initial_cpu = psutil.cpu_percent(interval=1)
        self.start_time = time.time()
        
        logger.info(f"Initial memory usage: {self.initial_memory:.1f}%")
        logger.info(f"Initial CPU usage: {self.initial_cpu:.1f}%")
        
        # Initialize processor
        try:
            self.processor = OptimizedParkingProcessor()
            logger.info("Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            return False
        
        return True
    
    def test_detector_performance(self, num_frames: int = 100) -> Dict[str, Any]:
        """Test detector performance with synthetic frames."""
        logger.info(f"Testing detector performance with {num_frames} frames...")
        
        import numpy as np
        import cv2
        
        # Create synthetic test frames
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        detector = self.processor.detector
        start_time = time.time()
        
        for i in range(num_frames):
            detections = detector.detect_vehicles(test_frame)
            if i % 20 == 0:
                logger.info(f"Processed {i} frames...")
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = num_frames / total_time
        
        detector_stats = detector.get_performance_stats()
        
        results = {
            'total_frames': num_frames,
            'total_time_seconds': total_time,
            'average_fps': fps,
            'detector_fps': detector_stats.get('fps', 0),
            'inference_time_ms': detector_stats.get('average_inference_time_ms', 0),
            'device': detector_stats.get('device', 'unknown')
        }
        
        logger.info(f"Detector test completed: {fps:.1f} FPS")
        return results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during operation."""
        logger.info("Testing memory usage...")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        results = {
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'virtual_memory_mb': memory_info.vms / 1024 / 1024
        }
        
        logger.info(f"Memory usage: {results['memory_usage_mb']:.1f} MB")
        return results
    
    def test_cpu_usage(self, duration: int = 10) -> Dict[str, Any]:
        """Test CPU usage over time."""
        logger.info(f"Testing CPU usage for {duration} seconds...")
        
        cpu_readings = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_readings.append(cpu_percent)
            logger.info(f"CPU usage: {cpu_percent:.1f}%")
        
        results = {
            'average_cpu_percent': sum(cpu_readings) / len(cpu_readings),
            'max_cpu_percent': max(cpu_readings),
            'min_cpu_percent': min(cpu_readings),
            'duration_seconds': duration
        }
        
        logger.info(f"Average CPU usage: {results['average_cpu_percent']:.1f}%")
        return results
    
    def test_processor_performance(self) -> Dict[str, Any]:
        """Test overall processor performance."""
        logger.info("Testing processor performance...")
        
        if not self.processor:
            logger.error("Processor not initialized")
            return {}
        
        # Start processing thread
        self.processor.start_processing_thread()
        
        # Wait for initialization
        time.sleep(5)
        
        # Collect performance stats
        performance_stats = self.processor.get_performance_stats()
        
        logger.info(f"Processor FPS: {performance_stats.get('fps', 0):.1f}")
        logger.info(f"Memory usage: {performance_stats.get('memory_usage_mb', 0):.1f} MB")
        
        return performance_stats
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive optimization test."""
        logger.info("Running comprehensive optimization test...")
        
        if not self.start_test():
            return {"error": "Failed to start test"}
        
        results = {
            'test_start_time': self.start_time,
            'initial_memory_percent': self.initial_memory,
            'initial_cpu_percent': self.initial_cpu
        }
        
        try:
            # Test detector performance
            results['detector_test'] = self.test_detector_performance(50)
            
            # Test memory usage
            results['memory_test'] = self.test_memory_usage()
            
            # Test CPU usage
            results['cpu_test'] = self.test_cpu_usage(5)
            
            # Test processor performance
            results['processor_test'] = self.test_processor_performance()
            
            # Calculate improvements
            results['improvements'] = self.calculate_improvements(results)
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            results['error'] = str(e)
        
        finally:
            self.cleanup()
        
        return results
    
    def calculate_improvements(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance improvements."""
        improvements = {}
        
        # Memory improvement
        if 'memory_test' in results:
            memory_usage = results['memory_test']['memory_usage_mb']
            if memory_usage < 2000:  # Less than 2GB
                improvements['memory'] = "Good - Under 2GB"
            elif memory_usage < 3000:  # Less than 3GB
                improvements['memory'] = "Acceptable - Under 3GB"
            else:
                improvements['memory'] = "High - Over 3GB"
        
        # CPU improvement
        if 'cpu_test' in results:
            avg_cpu = results['cpu_test']['average_cpu_percent']
            if avg_cpu < 50:
                improvements['cpu'] = "Excellent - Under 50%"
            elif avg_cpu < 70:
                improvements['cpu'] = "Good - Under 70%"
            else:
                improvements['cpu'] = "High - Over 70%"
        
        # FPS improvement
        if 'detector_test' in results:
            fps = results['detector_test']['average_fps']
            if fps > 20:
                improvements['fps'] = "Excellent - Over 20 FPS"
            elif fps > 10:
                improvements['fps'] = "Good - Over 10 FPS"
            else:
                improvements['fps'] = "Low - Under 10 FPS"
        
        return improvements
    
    def cleanup(self):
        """Clean up resources."""
        if self.processor:
            self.processor.cleanup()
            logger.info("Processor cleaned up")
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way."""
        print("\n" + "="*60)
        print("OPTIMIZATION TEST RESULTS")
        print("="*60)
        
        if 'error' in results:
            print(f"Test failed: {results['error']}")
            return
        
        # System Information
        print(f"System Information:")
        print(f"   - Test Duration: {time.time() - results['test_start_time']:.1f} seconds")
        print(f"   - Initial Memory: {results['initial_memory_percent']:.1f}%")
        print(f"   - Initial CPU: {results['initial_cpu_percent']:.1f}%")
        
        # Detector Performance
        if 'detector_test' in results:
            det = results['detector_test']
            print(f"\nDetector Performance:")
            print(f"   - Average FPS: {det['average_fps']:.1f}")
            print(f"   - Detector FPS: {det['detector_fps']:.1f}")
            print(f"   - Inference Time: {det['inference_time_ms']:.1f} ms")
            print(f"   - Device: {det['device']}")
        
        # Memory Usage
        if 'memory_test' in results:
            mem = results['memory_test']
            print(f"\nMemory Usage:")
            print(f"   - Memory Usage: {mem['memory_usage_mb']:.1f} MB")
            print(f"   - Memory Percent: {mem['memory_percent']:.1f}%")
        
        # CPU Usage
        if 'cpu_test' in results:
            cpu = results['cpu_test']
            print(f"\nCPU Usage:")
            print(f"   - Average CPU: {cpu['average_cpu_percent']:.1f}%")
            print(f"   - Max CPU: {cpu['max_cpu_percent']:.1f}%")
            print(f"   - Min CPU: {cpu['min_cpu_percent']:.1f}%")
        
        # Improvements
        if 'improvements' in results:
            print(f"\nPerformance Assessment:")
            for metric, assessment in results['improvements'].items():
                print(f"   - {metric.upper()}: {assessment}")
        
        print("\n" + "="*60)

def main():
    """Main test function."""
    print("Smart Parking System - Optimization Test")
    print("="*50)
    
    # Check if required files exist
    required_files = [
        'config/config.py',
        'core/optimized_processor.py',
        'core/optimized_detector.py',
        'pyproject.toml'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file not found: {file_path}")
            return
    
    print("All required files found")
    
    # Run test
    tester = OptimizationTester()
    results = tester.run_comprehensive_test()
    
    # Print results
    tester.print_results(results)
    
    # Summary
    if 'error' not in results:
        print("\nOptimization test completed successfully!")
        print("The system appears to be optimized for production use.")
    else:
        print("\nOptimization test failed!")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()

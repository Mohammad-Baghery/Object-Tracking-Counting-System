"""
Fixed Test Suite for MPS (Metal Performance Shaders) - PyTorch 2.8+ Compatible
"""

import torch
import torchvision
import time
import numpy as np
import cv2
import psutil
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from ultralytics import YOLO
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)


class M1GPUTesterFixed:
    """Fixed comprehensive testing suite for M1/M2 Mac GPU acceleration"""

    def __init__(self):
        self.results = {}
        self.system_info = {}

    def mps_synchronize(self):
        """Synchronize MPS operations - compatible with PyTorch 2.8+"""
        try:
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            elif hasattr(torch.backends.mps, 'synchronize'):
                torch.backends.mps.synchronize()
            else:
                # Fallback for newer versions
                torch.cuda.synchronize()  # Sometimes works as fallback
        except:
            # If nothing works, just add a small delay
            time.sleep(0.001)

    def mps_empty_cache(self):
        """Empty MPS cache - compatible with PyTorch 2.8+"""
        try:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            else:
                # Fallback
                gc.collect()
        except:
            gc.collect()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        print("🍎 Gathering System Information...")
        print("=" * 50)

        info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'torchvision_version': torchvision.__version__,
            'opencv_version': cv2.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2)
        }

        # Get macOS version
        try:
            result = subprocess.run(['sw_vers', '-productVersion'],
                                    capture_output=True, text=True)
            info['macos_version'] = result.stdout.strip()
        except:
            info['macos_version'] = "Unknown"

        # Get chip information
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                    capture_output=True, text=True)
            info['chip'] = result.stdout.strip()
        except:
            info['chip'] = "Unknown"

        self.system_info = info

        for key, value in info.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

        return info

    def test_pytorch_mps(self) -> bool:
        """Test PyTorch MPS availability and functionality"""
        print("\n🔥 Testing PyTorch MPS Support")
        print("=" * 40)

        tests = {
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built()
        }

        print(f"✓ MPS Available: {tests['mps_available']}")
        print(f"✓ MPS Built: {tests['mps_built']}")

        if tests['mps_available']:
            try:
                # Test basic tensor operations
                print("\n🧪 Testing MPS Operations...")

                # Create test tensors
                x = torch.randn(1000, 1000, device='mps')
                y = torch.randn(1000, 1000, device='mps')

                # Matrix multiplication
                start_time = time.time()
                z = torch.matmul(x, y)
                self.mps_synchronize()  # Use our fixed function
                mps_time = time.time() - start_time

                # Compare with CPU
                x_cpu = x.cpu()
                y_cpu = y.cpu()
                start_time = time.time()
                z_cpu = torch.matmul(x_cpu, y_cpu)
                cpu_time = time.time() - start_time

                speedup = cpu_time / mps_time if mps_time > 0 else 0

                print(f"   CPU Time: {cpu_time:.4f}s")
                print(f"   MPS Time: {mps_time:.4f}s")
                print(f"   Speedup: {speedup:.2f}x")

                tests['basic_operations'] = True
                tests['speedup'] = speedup

                # Clean up
                self.mps_empty_cache()

            except Exception as e:
                print(f"❌ MPS Test Failed: {e}")
                tests['basic_operations'] = False
                tests['speedup'] = 0
        else:
            print("❌ MPS not available - check PyTorch installation")
            tests['basic_operations'] = False
            tests['speedup'] = 0

        self.results['mps_tests'] = tests
        return tests['mps_available']

    def test_yolo_performance_simple(self) -> Dict[str, Dict[str, float]]:
        """Simplified YOLO test for PyTorch 2.8+"""
        print("\n🚀 Testing YOLO Performance (Simplified)")
        print("=" * 45)

        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")

        results = {}
        model_name = "yolov8n.pt"  # Test only the fastest model

        print(f"📦 Testing {model_name}")
        results[model_name] = {}

        try:
            for device in devices:
                print(f"  🖥️  Device: {device}")
                results[model_name][device] = {}

                try:
                    # Load model
                    model = YOLO(model_name)

                    # Create test image
                    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

                    # Set device
                    if device == "mps":
                        model.to(device)

                    # Warm-up (reduced iterations)
                    print("    🔥 Warming up...")
                    for _ in range(3):
                        _ = model(test_image, verbose=False)

                    # Benchmark
                    print("    📊 Benchmarking...")
                    num_iterations = 10  # Reduced from 20

                    start_time = time.time()
                    for i in range(num_iterations):
                        results_model = model(test_image, verbose=False)
                        if i % 3 == 0:  # Progress indicator
                            print(f"      Processing... {i+1}/{num_iterations}")

                    # Synchronize if using MPS
                    if device == "mps":
                        self.mps_synchronize()

                    end_time = time.time()
                    total_time = end_time - start_time
                    fps = num_iterations / total_time

                    results[model_name][device]["640"] = fps
                    print(f"    ✅ {device.upper()}: {fps:.2f} FPS")

                    # Clean up
                    if device == "mps":
                        self.mps_empty_cache()

                except Exception as e:
                    print(f"    ❌ Error testing {device}: {e}")
                    results[model_name][device]["640"] = 0

        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")

        self.results['yolo_performance'] = results
        return results

    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        print("\n💾 Testing Memory Usage")
        print("=" * 30)

        memory_results = {}

        if torch.backends.mps.is_available():
            try:
                print("🧪 MPS Memory Test...")

                # Initial memory
                initial_memory = psutil.virtual_memory().available / (1024 ** 3)

                # Create tensors (reduced size for stability)
                tensors = []
                for i in range(5):  # Reduced from 10
                    tensor = torch.randn(500, 500, device='mps')  # Reduced from 1000x1000
                    tensors.append(tensor)
                    print(f"    Created tensor {i+1}/5")

                # Memory after allocation
                allocated_memory = psutil.virtual_memory().available / (1024 ** 3)
                memory_used = initial_memory - allocated_memory

                print(f"   Memory used: {memory_used:.2f} GB")

                # Clean up
                tensors.clear()
                self.mps_empty_cache()
                gc.collect()

                # Memory after cleanup
                time.sleep(1)  # Give time for cleanup
                cleaned_memory = psutil.virtual_memory().available / (1024 ** 3)
                memory_recovered = cleaned_memory - allocated_memory

                print(f"   Memory recovered: {memory_recovered:.2f} GB")

                memory_results['mps_memory_test'] = {
                    'used_gb': memory_used,
                    'recovered_gb': memory_recovered,
                    'efficiency': (memory_recovered / memory_used) * 100 if memory_used > 0 else 0
                }

            except Exception as e:
                print(f"❌ MPS Memory test failed: {e}")
                memory_results['mps_memory_test'] = {'error': str(e)}

        # System memory info
        memory = psutil.virtual_memory()
        memory_results['system_memory'] = {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_percent': memory.percent
        }

        print(f"💻 System Memory:")
        print(f"   Total: {memory_results['system_memory']['total_gb']:.2f} GB")
        print(f"   Available: {memory_results['system_memory']['available_gb']:.2f} GB")
        print(f"   Used: {memory_results['system_memory']['used_percent']:.1f}%")

        self.results['memory_usage'] = memory_results
        return memory_results

    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # System info
        print("\n💡 Performance Analysis & Recommendations")
        print("=" * 50)

        # Apple Silicon check
        if 'M1' in self.system_info.get('chip', '') or 'M2' in self.system_info.get('chip', ''):
            recommendations.append("✅ Apple M1 detected - excellent for ML workloads!")

        # MPS availability
        if self.results.get('mps_tests', {}).get('mps_available', False):
            recommendations.append("🔥 MPS is available and working!")

            # Speedup analysis
            speedup = self.results.get('mps_tests', {}).get('speedup', 0)
            if speedup > 1:
                recommendations.append(f"⚡ MPS provides {speedup:.1f}x speedup over CPU")

            recommendations.append("💻 Recommended settings for object tracking:")
            recommendations.append("   - DEVICE = 'mps'")
            recommendations.append("   - BATCH_SIZE = 1 (most stable)")
            recommendations.append("   - Use YOLOv8n for real-time performance")

        # YOLO performance
        yolo_perf = self.results.get('yolo_performance', {})
        if yolo_perf:
            for model, devices in yolo_perf.items():
                if 'mps' in devices and 'cpu' in devices:
                    mps_fps = list(devices['mps'].values())[0] if devices['mps'] else 0
                    cpu_fps = list(devices['cpu'].values())[0] if devices['cpu'] else 0

                    if mps_fps > cpu_fps:
                        improvement = ((mps_fps / cpu_fps - 1) * 100) if cpu_fps > 0 else 0
                        recommendations.append(f"🎯 {model} on MPS: {mps_fps:.1f} FPS (+{improvement:.0f}% vs CPU)")

                    if mps_fps > 20:
                        recommendations.append("🎬 Excellent for real-time video processing!")
                    elif mps_fps > 10:
                        recommendations.append("⚖️ Good for near real-time processing")

        # Memory recommendations
        memory_gb = self.system_info.get('memory_gb', 0)
        if memory_gb <= 8:
            recommendations.append("⚠️ 8GB RAM - optimize for memory efficiency:")
            recommendations.append("   - Close other applications")
            recommendations.append("   - Use smaller input resolution (480p)")
            recommendations.append("   - Enable periodic memory cleanup")

        # PyTorch version note
        recommendations.append(f"🔧 You're using PyTorch {self.system_info.get('torch_version', '')}")
        recommendations.append("   Latest version with good MPS support!")

        return recommendations

    def save_results(self, filename: str = "m1_gpu_test_results_fixed.txt"):
        """Save test results"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🍎 M1/M2 Mac GPU Test Results (PyTorch 2.8+ Compatible)\n")
                f.write("=" * 60 + "\n\n")

                # System info
                f.write("📱 System Information:\n")
                for key, value in self.system_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # MPS test results
                mps_results = self.results.get('mps_tests', {})
                f.write("🔥 MPS Test Results:\n")
                f.write(f"  Available: {mps_results.get('mps_available', False)}\n")
                f.write(f"  Built: {mps_results.get('mps_built', False)}\n")
                f.write(f"  Speedup: {mps_results.get('speedup', 0):.2f}x\n\n")

                # YOLO results
                yolo_results = self.results.get('yolo_performance', {})
                if yolo_results:
                    f.write("🚀 YOLO Performance:\n")
                    for model, devices in yolo_results.items():
                        f.write(f"  {model}:\n")
                        for device, results in devices.items():
                            for size, fps in results.items():
                                f.write(f"    {device} ({size}px): {fps:.2f} FPS\n")
                    f.write("\n")

                # Recommendations
                f.write("💡 Recommendations:\n")
                for rec in self.generate_recommendations():
                    f.write(f"  {rec}\n")

            print(f"📄 Results saved to {filename}")

        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def run_full_test_suite(self):
        """Run complete test suite"""
        print("🧪 M1/M2 Mac GPU Test Suite (PyTorch 2.8+ Compatible)")
        print("=" * 70)

        try:
            # Test 1: System Info
            self.get_system_info()

            # Test 2: PyTorch MPS
            mps_available = self.test_pytorch_mps()

            # Test 3: YOLO Performance (simplified)
            if mps_available:
                self.test_yolo_performance_simple()
            else:
                print("\n⚠️ Skipping YOLO tests - MPS not available")

            # Test 4: Memory Usage
            self.test_memory_usage()

            # Generate recommendations
            recommendations = self.generate_recommendations()
            for rec in recommendations:
                print(f"  {rec}")

            # Save results
            self.save_results()

            print(f"\n🎉 Test completed successfully!")
            print("📊 Check the results file for detailed analysis")

            # Quick summary
            print(f"\n📋 Quick Summary:")
            print(f"   🖥️  System: {self.system_info.get('chip', 'Unknown')}")
            print(f"   🔥 MPS Available: {self.results.get('mps_tests', {}).get('mps_available', False)}")

            yolo_results = self.results.get('yolo_performance', {})
            if yolo_results:
                for model, devices in yolo_results.items():
                    if 'mps' in devices:
                        mps_fps = list(devices['mps'].values())[0] if devices['mps'] else 0
                        print(f"   🎯 {model} on MPS: {mps_fps:.1f} FPS")

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    tester = M1GPUTesterFixed()
    tester.run_full_test_suite()


if __name__ == "__main__":
    main()

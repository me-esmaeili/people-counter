import cv2
import time
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue

from ultralytics import YOLO
# For GPU stats - install with pip install gputil
try:
    import GPUtil

    gputil_available = True
except ImportError:
    gputil_available = False
    print("GPUtil not found. Install with: pip install gputil")


class CameraSimulator:
    """Simulates multiple camera streams"""

    def __init__(self, num_cameras, resolution=(640, 640)):
        self.num_cameras = num_cameras
        self.resolution = resolution
        self.queues = [Queue(maxsize=30) for _ in range(num_cameras)]
        self.stopped = False

    def start(self):
        # Start a thread for each camera
        self.threads = []
        for i in range(self.num_cameras):
            thread = Thread(target=self.update, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self

    def update(self, camera_id):
        # Simulate camera frame capture
        while not self.stopped:
            # Generate a random frame (or load a sample image for more realistic testing)
            frame = np.random.randint(0, 255, (self.resolution[0], self.resolution[1], 3),
                                      dtype=np.uint8)

            # Put frame in queue if there's room
            if not self.queues[camera_id].full():
                self.queues[camera_id].put(frame)
            time.sleep(0.01)  # Simulate 100fps camera

    def read_batch(self, batch_size=None):
        """Read one frame from each camera, up to batch_size"""
        if batch_size is None:
            batch_size = self.num_cameras

        batch = []
        for i in range(min(batch_size, self.num_cameras)):
            if not self.queues[i].empty():
                frame = self.queues[i].get()
                batch.append(frame)

        return batch if batch else None

    def stop(self):
        self.stopped = True
        for thread in self.threads:
            thread.join()


def get_system_stats():
    """Get GPU and system statistics"""
    stats = {
        "cpu_utilization": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent
    }

    if gputil_available:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get the first GPU
            stats.update({
                "gpu_memory_used": gpu.memoryUsed,
                "gpu_memory_total": gpu.memoryTotal,
                "gpu_utilization": gpu.load * 100
            })

    return stats


def benchmark_yolo(model, num_cameras, resolution=(640, 640),
                   batch_size=4, duration=30, conf_threshold=0.25):
    """
    Benchmark YOLO performance with specified number of cameras

    Args:
        model: The YOLO model
        num_cameras: Number of cameras to simulate
        resolution: Input resolution
        batch_size: Processing batch size
        duration: Test duration in seconds
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary with benchmark results
    """
    # Start camera simulators
    camera_sim = CameraSimulator(num_cameras, resolution).start()

    frames_processed = 0
    detections = 0
    start_time = time.time()
    end_time = start_time + duration

    stats_history = []

    # Warmup
    print("Warming up model...")
    dummy_input = torch.zeros((1, 3, resolution[0], resolution[1])).to(model.device)
    for _ in range(10):
        _ = model(dummy_input)

    print(f"Running benchmark with {num_cameras} cameras for {duration} seconds...")

    try:
        while time.time() < end_time:
            # Get batch of frames
            batch = camera_sim.read_batch(batch_size)
            if not batch:
                time.sleep(0.001)
                continue

            # Perform inference
            results = model(batch)

            # Count detections (if using a real YOLO model)
            try:
                for result in results.xyxy:  # For YOLOv5
                    detections += len(result[result[:, 4] > conf_threshold])
            except:
                # If using a different YOLO version, this might fail
                pass

            # Record system stats periodically
            if frames_processed % 100 == 0:
                stats_history.append(get_system_stats())

            # Update counter
            frames_processed += len(batch)

    except Exception as e:
        print(f"Error during benchmarking: {e}")
    finally:
        # Calculate statistics
        camera_sim.stop()

        total_time = time.time() - start_time
        fps = frames_processed / total_time

        # Calculate average system stats
        if stats_history:
            avg_stats = {k: sum(d.get(k, 0) for d in stats_history) / len(stats_history)
                         for k in stats_history[0].keys()}
        else:
            avg_stats = {}

        result = {
            "num_cameras": num_cameras,
            "fps": fps,
            "fps_per_camera": fps / num_cameras if num_cameras > 0 else 0,
            "total_frames": frames_processed,
            "total_detections": detections,
            "total_time": total_time,
            **avg_stats
        }

        return result


def create_benchmark_report(results):
    """Create a detailed benchmark report without matplotlib dependency issues"""
    # Print text report
    print("\n====== YOLO CAMERA SCALING BENCHMARK REPORT ======")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # System information
    if gputil_available:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"\nGPU: {gpu.name}")
            print(f"GPU Memory: {gpu.memoryTotal} MB")

    print(f"\nCPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"RAM: {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")

    # Results table
    print("\n--- Performance Results ---")
    print(f"{'Cameras':<10} {'Total FPS':<10} {'FPS/Camera':<12} {'GPU Mem %':<10} {'GPU Util %':<10}")

    for r in results:
        gpu_mem_percent = r.get('gpu_memory_used', 0) / r.get('gpu_memory_total',
                                                              1) * 100 if 'gpu_memory_used' in r else 0
        gpu_util = r.get('gpu_utilization', 0)

        print(f"{r['num_cameras']:<10} {r['fps']:<10.2f} {r['fps_per_camera']:<12.2f} "
              f"{gpu_mem_percent:<10.1f} {gpu_util:<10.1f}")

    # Recommendation
    recommended_cameras = 0
    for r in results:
        if r['fps_per_camera'] >= 15:  # Assuming 15 FPS is minimum acceptable
            recommended_cameras = r['num_cameras']
        else:
            break

    print(f"\nRecommended maximum cameras: {recommended_cameras} "
          f"(maintaining at least 15 FPS per camera)")

    # Save results to CSV file as an alternative to visualization
    try:
        import csv
        with open('yolo_camera_benchmark_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['num_cameras', 'fps', 'fps_per_camera', 'gpu_memory_used',
                          'gpu_memory_total', 'gpu_utilization']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for r in results:
                # Filter only the fields we want
                row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(row)

        print("\nBenchmark results saved as 'yolo_camera_benchmark_results.csv'")
    except Exception as e:
        print(f"Could not save CSV file: {e}")

    # Try to generate visualization with explicit backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Plot FPS vs. Camera Count
        plt.figure(figsize=(12, 10))

        # Total FPS
        plt.subplot(2, 2, 1)
        plt.plot([r['num_cameras'] for r in results],
                 [r['fps'] for r in results], 'bo-', linewidth=2)
        plt.xlabel('Number of Cameras')
        plt.ylabel('Total FPS')
        plt.title('Total Processing Rate')
        plt.grid(True)

        # FPS per Camera
        plt.subplot(2, 2, 2)
        plt.plot([r['num_cameras'] for r in results],
                 [r['fps_per_camera'] for r in results], 'ro-', linewidth=2)
        plt.axhline(y=15, color='g', linestyle='--', label='15 FPS Threshold')
        plt.xlabel('Number of Cameras')
        plt.ylabel('FPS per Camera')
        plt.title('Per-Camera Processing Rate')
        plt.grid(True)
        plt.legend()

        # GPU Memory Usage
        if 'gpu_memory_used' in results[0]:
            plt.subplot(2, 2, 3)
            plt.plot([r['num_cameras'] for r in results],
                     [r['gpu_memory_used'] / r['gpu_memory_total'] * 100 for r in results], 'go-', linewidth=2)
            plt.xlabel('Number of Cameras')
            plt.ylabel('GPU Memory Usage (%)')
            plt.title('GPU Memory Utilization')
            plt.grid(True)

        # GPU Utilization
        if 'gpu_utilization' in results[0]:
            plt.subplot(2, 2, 4)
            plt.plot([r['num_cameras'] for r in results],
                     [r['gpu_utilization'] for r in results], 'mo-', linewidth=2)
            plt.xlabel('Number of Cameras')
            plt.ylabel('GPU Utilization (%)')
            plt.title('GPU Compute Utilization')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('yolo_camera_benchmark.png')
        print("\nBenchmark visualization saved as 'yolo_camera_benchmark.png'")
    except Exception as e:
        print(f"Note: Could not generate visualization: {e}")
        print("This is non-critical - the CSV and text report contain all the data.")

def main():
    """Main benchmark function"""
    # Load your existing YOLO model
    # This example uses YOLOv5, but adapt it to your model
    print("Loading YOLO model...")
    try:
        # Try to load using the ultralytics package (YOLOv5/v8)
        import torch

        # Use YOLOv5 - make sure to install with: pip install ultralytics
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

        # Uncomment below for YOLOv8 if that's what you're using
        # from ultralytics import YOLO
        model = YOLO("assets/yolo11n.pt")

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nIf you're using a custom YOLO model, please modify this script to load your model.")
        return

    # Camera counts to test
    camera_counts = [1, 2, 4, 8, 16]

    # Run benchmarks
    results = []
    for count in camera_counts:
        try:
            result = benchmark_yolo(model, count)
            results.append(result)
            print(f"Results for {count} cameras: {result['fps']:.2f} total FPS, "
                  f"{result['fps_per_camera']:.2f} FPS/camera")

            # Stop if performance is too low
            if result['fps_per_camera'] < 5:
                print("Performance threshold reached, stopping benchmark")
                break
        except RuntimeError as e:
            print(f"Failed at {count} cameras, likely due to GPU memory limitations: {e}")
            break
        except Exception as e:
            print(f"Error during benchmark with {count} cameras: {e}")
            break

    # Generate report
    if results:
        create_benchmark_report(results)


if __name__ == "__main__":
    main()
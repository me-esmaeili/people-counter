import platform
import psutil
from datetime import datetime
from threading import Thread
import time
import os

# Only import vcgencmd on Raspberry Pi (Linux-based systems with vcgencmd available)
if platform.system() == "Linux" and os.path.exists("/usr/bin/vcgencmd"):
    try:
        from vcgencmd import Vcgencmd
        VCGENCMD_AVAILABLE = True
    except ImportError:
        VCGENCMD_AVAILABLE = False
else:
    VCGENCMD_AVAILABLE = False

class SystemLogger:
    """Handles system monitoring and logging, compatible with Windows and Raspberry Pi"""
    def __init__(self, log_file="readings.txt"):
        self.log_file = log_file
        if VCGENCMD_AVAILABLE:
            self.vcgm = Vcgencmd()
        self._initialize_log()

    def _initialize_log(self):
        """Initialize the log file with headers"""
        with open(self.log_file, "a+") as fb:
            fb.write("Date,Time,Temperature (°C),Cpu_usage (%),Memory_usage(%)\n")

    def start(self):
        """Start the logging thread"""
        self.thread = Thread(target=self.log)
        self.thread.start()

    def _get_temperature(self):
        """Get system temperature, platform-dependent"""
        if VCGENCMD_AVAILABLE:
            return self.vcgm.measure_temp()
        else:
            try:
                if platform.system() == "Windows":
                    import wmi
                    w = wmi.WMI(namespace="root\\wmi")
                    temp = w.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature / 10.0 - 273.15
                    return temp
            except (ImportError, Exception):
                return "N/A"

    def log(self):
        """Run the logging loop"""
        while True:
            temp = self._get_temperature()
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            now = datetime.now()
            # Use %Y instead of %-Y for cross-platform compatibility
            log_string = f'{now.strftime("%A %m %Y")},{now.strftime("%H:%M:%S")},{temp},{cpu_usage},{mem_usage}\n'
            with open(self.log_file, "a+") as fb:
                fb.write(log_string)
            time.sleep(1)

if __name__ == "__main__":
    logger = SystemLogger()
    logger.start()
    time.sleep(5)
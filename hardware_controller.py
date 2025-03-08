import platform
import time
from multiprocessing import Process

# Only import GPIO on Raspberry Pi (Linux-based systems)
if platform.system() == "Linux":
    try:
        import RPi.GPIO as GPIO

        GPIO_AVAILABLE = True
    except ImportError:
        GPIO_AVAILABLE = False
else:
    GPIO_AVAILABLE = False


class HardwareController:
    """Manages hardware components like LED, compatible with Windows and Raspberry Pi"""

    def __init__(self, pin=8):
        self.pin = pin
        self.led_state = False

        if GPIO_AVAILABLE:
            # Raspberry Pi initialization
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        else:
            # Windows or non-GPIO system initialization
            print("GPIO not available. Running in simulation mode.")

    def start(self):
        """Start the hardware controller in a separate process"""
        self.thread = Process(target=self.run)
        self.thread.start()

    def run(self):
        """Run the LED control loop"""
        while True:
            time.sleep(1)
            if GPIO_AVAILABLE:
                # Raspberry Pi: Control actual GPIO pin
                GPIO.output(self.pin, GPIO.HIGH if self.led_state else GPIO.LOW)
            else:
                pass
                # Windows: Simulate LED state change with print
                # print(f"Simulated LED {'ON' if self.led_state else 'OFF'} on pin {self.pin}")

    def set_led(self, state):
        """Set the LED state"""
        self.led_state = state

    def cleanup(self):
        """Cleanup GPIO resources (only on Raspberry Pi)"""
        if GPIO_AVAILABLE:
            GPIO.cleanup()


if __name__ == "__main__":
    # Test the controller
    controller = HardwareController()
    controller.start()

    # Simulate some LED changes
    controller.set_led(True)
    time.sleep(2)
    controller.set_led(False)
    time.sleep(2)
    controller.cleanup()
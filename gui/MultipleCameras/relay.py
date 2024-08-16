import logging
import threading
from datetime import datetime
import time
import multiprocessing as mp

import serial

logger = logging.getLogger("AutoInspection")


class Relay:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Relay, cls).__new__(cls)
                cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        # Set the serial port and baud rate
        self.serial_port = 'COM6'  # Change this to match your Arduino's serial port
        self.baud_rate = 9600
        self.timeout = 1
        self.relay = None
        self.__initialized = True
        self.num_retry = 0
        self.max_num_retry = 10

        self.response_event = [mp.Event(), mp.Event()]

    def open(self):
        try:
            self.num_retry += 1
            if self.relay is None or not self.relay.is_open:
                # Initialize serial communication
                self.relay = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
                logger.info(f"Opened serial port {self.serial_port} at {self.baud_rate} baud.")
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.serial_port}: {e}")

    def close(self):
        if self.relay and self.relay.is_open:
            try:
                # Close the serial connection when done
                self.relay.close()
                logger.info(f"Closed serial port {self.serial_port}.")
            except serial.SerialException as e:
                logger.error(f"Failed to close serial port {self.serial_port}: {e}")

    def _write_to_serial(self, index: int, delay: int, times=4):
        try:
            sign_str = str(index) * times
            # sign_str = f"{index}-{delay}="
            logger.info(f"Pushing the camera {index}")
            self.response_event[index].clear()
            self.relay.write(sign_str.encode())
            self.response_event[index].set()
            logger.info(f"Successfully wrote to serial: {sign_str}")
        except serial.SerialException as e:
            logger.error(f"Failed to write to serial: {e}")
        except Exception as e:
            logger.error(f"An error occurred while writing to serial: {e}")

    def get_response(self, index: int):
        return self.response_event[index]

    def write(self, index: int, delay: int = 0, times=4):
        if self.relay and self.relay.is_open:
            self._write_to_serial(index, delay, times=times)
            logger.info(f"Write operation at {datetime.now()}")
        else:
            logger.error("Cannot write to serial. The serial port is not open.")
            self.close()
            self.open()


if __name__ == "__main__":
    relay = Relay()
    relay.open()
    while True:
        relay.write(0)
        time.sleep(0.2)

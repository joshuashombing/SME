import serial


class Relay:

    def __init__(self):
        # Set the serial port and baud rate
        self.serial_port = '/COM3'  # Change this to match your Arduino's serial port
        self.baud_rate = 9600
        self.timeout = 1
        self.relay = None

    def open(self):
        # Initialize serial communication
        self.relay = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)

    def close(self):
        # Close the serial connection when done
        self.relay.close()

    def write(self, result: str):
        # Send result to the serial connection
        self.relay.write(result.encode())

# This script tests if the power source is connected through the rs485 converter.

import minimalmodbus
import serial

def test_power_source_connection(port, slave_address):
    """
    Test connection to RS485 power source.

    Args:
        port (str): Serial port (e.g., '/dev/ttyUSB0' or 'COM3').
        slave_address (int): Modbus slave address of the power source.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        # Create an instrument instance
        instrument = minimalmodbus.Instrument(port=port, slaveaddress=slave_address, mode=minimalmodbus.MODE_RTU)

        # Configure serial settings
        instrument.serial.baudrate = 9600  # Match your device's baud rate
        instrument.serial.bytesize = 8
        instrument.serial.parity = serial.PARITY_NONE
        instrument.serial.stopbits = 1
        instrument.serial.timeout = 1  # Set timeout for communication

        # Clear buffers before communication
        instrument.clear_buffers_before_each_transaction = True

        print(f"Testing connection on {port} with slave address {slave_address}...")

        # Send a simple query (e.g., read a register to test communication)
        # Register address and function code depend on your device's Modbus implementation.
        response = instrument.read_register(registeraddress=0x1000, functioncode=3)  # Replace with valid register
        print(f"Received response: {response}")
        print("Connection successful!")
        return True

    except minimalmodbus.NoResponseError:
        print("No response from the power source. Check connections or power.")
    except minimalmodbus.ModbusException as e:
        print(f"Modbus communication error: {e}")
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False

if __name__ == "__main__":
    # Replace with your actual serial port and slave address
    port = '/dev/tty.usbserial-B003LY1G'  # e.g., '/dev/ttyUSB0' or 'COM3'
    slave_address = 1  # Adjust based on your device configuration
    success = test_power_source_connection(port, slave_address)
    if success:
        print("Power source is communicating with the computer!!!!!!!!!!!!!!")
    else:
        print("Failed to communicate with the power source.")

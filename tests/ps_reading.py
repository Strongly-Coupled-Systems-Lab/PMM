# This script is for the power supplies and tests if the RS485 converter is working (i.e. TXD and RXD both flash)

import serial
import time

def test_rs485(port, baudrate, message):
    try:
        # Open the serial port
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"Opened {port} successfully.")

        while True:
            # Send data
            ser.write(message.encode('utf-8'))
            print(f"Sent: {message}")
            
            # Wait briefly to allow receiver to process
            time.sleep(0.1)

            # Read the echoed data (loopback or other device response)
            if ser.in_waiting:
                response = ser.read_all().decode('utf-8', errors='ignore')
                print(f"Received: {response}")
            
            # Pause between iterations to keep lights flashing distinctly
            time.sleep(1)

    except serial.SerialException as e:
        print(f"Error opening or using the serial port: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print(f"Closed {port}.")

if __name__ == "__main__":
    # Replace '/dev/ttyUSB0' or 'COM3' with your RS485 port name
    test_rs485(port='/dev/tty.usbserial-B003LKT9', baudrate=9600, message="RS485 Test")

# to terminate: ctrl+c in terminal

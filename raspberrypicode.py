import RPi.GPIO as GPIO
import socket
import time
import threading

# 🚗 Motor Configuration
MOTOR1_PIN = 18
MOTOR2_PIN = 13  # 🆕 Added second motor
DEFAULT_SPEED = 30  # Default speed (30% duty cycle)

# 🛠 GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR1_PIN, GPIO.OUT)
GPIO.setup(MOTOR2_PIN, GPIO.OUT)  # 🆕 Setup second motor

# 🌟 PWM Setup for Motor Speed Control
pwm1 = GPIO.PWM(MOTOR1_PIN, 100)  # 100 Hz frequency
pwm2 = GPIO.PWM(MOTOR2_PIN, 100)  # 🆕 Added PWM for second motor
pwm1.start(DEFAULT_SPEED)  # Start first motor at default speed
pwm2.start(DEFAULT_SPEED)  # 🆕 Start second motor at default speed

# 🌐 UDP Socket Setup
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 🔄 Shared Variables
current_speed = DEFAULT_SPEED
target_speed = DEFAULT_SPEED
last_sign_detected_time = time.time()  # Track last sign detection
selected_mode = "Assistance"

# 📂 Load Mode from File
try:
    with open("mode.txt", "r") as file:
        selected_mode = file.read().strip()
    print(f"✅ Loaded Mode: {selected_mode}")
except FileNotFoundError:
    print("⚠ Mode file not found. Using default: Assistance")

# 🚀 Function to Change Speed Gradually
def gradual_speed_change(new_speed, duration=3):
    global current_speed
    steps = 10
    step_size = (new_speed - current_speed) / steps
    interval = duration / steps

    for _ in range(steps):
        current_speed += step_size
        pwm1.ChangeDutyCycle(max(0, min(100, current_speed)))  # Ensure within 0-100%
        pwm2.ChangeDutyCycle(max(0, min(100, current_speed)))  # 🆕 Sync second motor speed
        print(f"🔄 Speed Updating: {int(current_speed)}%")
        time.sleep(interval)

# ⏳ Reset Speed to Default After 10 Seconds if No Sign Detected
def reset_speed_monitor():
    global last_sign_detected_time, target_speed
    while True:
        time.sleep(1)
        if time.time() - last_sign_detected_time > 10:  # If 10 sec passed since last detection
            if current_speed != DEFAULT_SPEED:
                print("⏳ No new sign detected. Resetting to default speed...")
                target_speed = DEFAULT_SPEED
                gradual_speed_change(DEFAULT_SPEED, duration=3)

# 🚦 Process Received Commands from Windows
def process_command(command):
    global target_speed, last_sign_detected_time, selected_mode

    print(f"📩 Received from Windows: {command}")

    if "Mode:" in command:
        selected_mode = command.split(":")[1].strip()
        with open("mode.txt", "w") as file:
            file.write(selected_mode)
        print(f"🔄 Mode changed to: {selected_mode}")
        return

    # Update last sign detected time
    last_sign_detected_time = time.time()

    # Assistance Mode → Only Display Sign, No Motor Control
    if selected_mode == "Assistance":
        print(f"🔵 Assistance Mode: Detected Sign - {command}")  # ✅ FIXED: Now displays detected sign!
        return

    # Autonomous Mode → Adjust Speed Based on Sign
    if "SpeedLimit" in command:
        speed_value = int(command.split(":")[1])

        if speed_value < current_speed:
            print(f"⚠ Speed Limit {speed_value}% detected. Slowing down...")
            target_speed = speed_value
            gradual_speed_change(speed_value, duration=3)
        else:
            print(f"✅ Speed Limit {speed_value}% detected, but maintaining current speed at {current_speed}%.")

    elif command == "Stop":
        print("🛑 Stop sign detected. Stopping motor...")
        target_speed = 0
        gradual_speed_change(0, duration=2)

# 🌍 Start Listening for UDP Messages
def receive_commands():
    while True:
        data, addr = sock.recvfrom(1024)
        process_command(data.decode())

# 🔥 Start Threads
threading.Thread(target=reset_speed_monitor, daemon=True).start()
threading.Thread(target=receive_commands, daemon=True).start()

# 💤 Keep Script Running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("⚠ Stopping motor and exiting...")
    pwm1.stop()
    pwm2.stop()  # 🆕 Stop second motor
    GPIO.cleanup()
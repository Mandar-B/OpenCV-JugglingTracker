import cv2
import numpy as np
import time

# Open your computer's webcam (you may need to specify a different camera index)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to another index if needed

# Define parameters
feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=25, blockSize=9)

# Screen height and threshold
screen_height = 480  # Adjust this value to match your video resolution
threshold_y = screen_height - 20  # Adjust the threshold as needed

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Create a list of xy coordinates of detected juggling balls
points = []

# Count of balls entering the screen
balls_entered = 0

# Additional counter
additional_counter = 0

# Time variables
previous_time = time.time()
previous_balls_entered = 0
check_interval = 0.8  # Check interval in seconds
increase_threshold = 13  # Threshold for increase in balls_entered

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)  # Subtract background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Clean up image

    # Convert the frame to HSV for tennis ball color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for tennis ball color (adjust as needed)
    lower_tennis_ball = np.array([20, 100, 100], dtype=np.uint8)
    upper_tennis_ball = np.array([40, 255, 255], dtype=np.uint8)

    # Create a binary mask for tennis ball color
    tennis_ball_mask = cv2.inRange(hsv_frame, lower_tennis_ball, upper_tennis_ball)

    # Combine the tennis ball color mask with the motion mask
    fgmask = cv2.bitwise_and(fgmask, tennis_ball_mask)

    detected_balls = cv2.goodFeaturesToTrack(fgmask, mask=None, **feature_params)  # Find balls

    if detected_balls is not None:
        detected_balls = np.int0(detected_balls)

        for ball in detected_balls:
            x, y = ball.ravel()

            # Check if the ball crosses the threshold (enters the screen)
            if y >= threshold_y:
                balls_entered += 1

            # Draw a circle around the ball
            cv2.circle(frame, (x, y), 25, (255, 0, 0), 2)
            points.append((x, y))

    # Calculate the time elapsed since the previous frame
    current_time = time.time()
    time_elapsed = current_time - previous_time


    # Check for the condition to increment the additional counter
    if time_elapsed >= check_interval:
        if (balls_entered - previous_balls_entered) > increase_threshold:
            additional_counter += 1
        previous_time = current_time
        previous_balls_entered = balls_entered



    # Display the total balls entered and the additional counter on the frame
    text = f'Balls Entered: {balls_entered}, Additional Counter: {additional_counter}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Live Feed', frame)  # Show the live feed
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the live feed
        break

cap.release()
cv2.destroyAllWindows()

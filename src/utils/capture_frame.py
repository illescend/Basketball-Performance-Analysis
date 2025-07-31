import cv2
import os

# --- Configuration ---
# TODO: Set these three variables before running

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIDEO_INPUT_DIR = os.path.join(BASE_DIR,"data", "footage")
# 1. Path to your video file
VIDEO_PATH = os.path.join(VIDEO_INPUT_DIR,"batch10.mp4")

# 2. The exact frame number you want to capture
FRAME_TO_CAPTURE = 1858 # Example: The frame of your hand clap for sync

# 3. The name for the output image file
OUTPUT_FILENAME = "batch10_ball.png"


# --- Main Script ---

def capture_specific_frame(video_path, frame_number, output_path):
    """
    Opens a video, seeks to a specific frame, and saves it as an image.

    Args:
        video_path (str): The full path to the video file.
        frame_number (int): The frame number to capture (0-indexed).
        output_path (str): The path to save the output PNG image.
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{os.path.abspath(video_path)}'")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        print("Please check the path and ensure OpenCV has the correct codecs.")
        return

    # Get total number of frames to validate the requested frame number
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"Error: Requested frame {frame_number} is out of bounds.")
        print(f"The video only has {total_frames} frames (0 to {total_frames - 1}).")
        cap.release()
        return

    # Set the video's current position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()

    # Check if the frame was read successfully
    if not success:
        print(f"Error: Could not read frame {frame_number} from the video.")
        cap.release()
        return

    # Save the captured frame as an image
    try:
        cv2.imwrite(output_path, frame)
        print(f"Successfully captured frame {frame_number} and saved it to:")
        print(f"--> {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error: Could not save the image. Reason: {e}")

    # Release the video capture object
    cap.release()


if __name__ == '__main__':
    # Ensure the output directory exists if a path is specified
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    capture_specific_frame(VIDEO_PATH, FRAME_TO_CAPTURE, OUTPUT_FILENAME)
import os
from pathlib import Path
import cv2
import datetime
import numpy as np
import pandas as pd

step_by_step = False

threshhold = 90
monitor_mask_on_threshhold = 80
monitor_mask_off_threshhold = 10
led_mask_on_threshhold = 35
led_mask_off_threshhold = 5
show_image_size = (1000, 750)
# colours should be entered in RGB format, for better understanding
# these bounds are subsequently converted to the OpenCV-needed BGR-format
led_lower_bound = np.array([183, 38, 8])
led_upper_bound = np.array([255, 255, 255])
monitor_lower_bound = np.array([0, 19, 13])
monitor_upper_bound = np.array([56, 166, 129])

led_lower_bound = led_lower_bound[::-1]
led_upper_bound = led_upper_bound[::-1]
monitor_lower_bound = monitor_lower_bound[::-1]
monitor_upper_bound = monitor_upper_bound[::-1]

export_path = ".\\data"
videos_path = ".\\videos"

# video = cv2.VideoCapture("./videos/Test/test_godot.mp4")
videos = []

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def check_if_path_exists_or_create(path):
    try:
        existing = os.path.exists(path)

        if not existing:
            os.makedirs(path)
    except:
        print("Creating didn't work")
        return False

    return True


check_if_path_exists_or_create(videos_path)

for root, dirs, files in os.walk(videos_path):
    for file in files:
        if files.__sizeof__ != 0:
            print("File Found:", f"{root}\\{file}")
            videos.append(f"{root}\\{file}")


def create_csv_from_df(video_name, led_frames, monitor_frames):
    d = dict(leds=led_frames, monitor=monitor_frames)

    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()])).to_csv(
        f"{export_path}\\{video_name}_data_{timestamp}.csv", index=False
    )

    # frame_data.to_csv(f"{export_path}\\{video_name}_data_{timestamp}.csv", index=False)


def show_image_with_size(name, image, size):
    resized = cv2.resize(image, size)

    cv2.imshow(name, resized)
    pass


def crop_image_by_percent(image, height_percent=100, width_percent=100):
    height = int(image.shape[0] * (height_percent / 100))
    width = int(image.shape[1] * (width_percent / 100))

    cropped = image[0:height, 0:width]

    return cropped


def get_percentage_of_mask_fill(frame):
    uniques, counts = np.unique(frame, return_counts=True)

    fill_percantage = 0

    if counts.size >= 2:
        fill_percantage = counts[1] / np.sum(counts) * 100

    return fill_percantage


def get_monitor_mask_percentage(frame):
    lower = monitor_lower_bound
    upper = monitor_upper_bound

    monitor_mask = cv2.inRange(frame, lower, upper)
    monitor_mask = monitor_mask[
        int(monitor_mask.shape[0] * 0) : int(monitor_mask.shape[0] * 0.7),
        int((monitor_mask.shape[1]) * 0) : int((monitor_mask.shape[1]) * 1),
    ]

    monitor_mask_percentage = get_percentage_of_mask_fill(monitor_mask)

    # show_image_with_size("monitor_mask", monitor_mask, show_image_size)

    return monitor_mask, monitor_mask_percentage


def get_led_mask_percentage(frame):
    lower = led_lower_bound
    upper = led_upper_bound

    led_mask = cv2.inRange(frame, lower, upper)
    led_mask = led_mask[
        int(led_mask.shape[0] * 0.75) : int(led_mask.shape[0] * 0.85),
        int((led_mask.shape[1] / 2) * 0.95) : int((led_mask.shape[1] / 2) * 1.05),
    ]
    led_mask_percentage = get_percentage_of_mask_fill(led_mask)

    # show_image_with_size("led_mask", led_mask, show_image_size)

    return led_mask, led_mask_percentage


def process_video(video, video_name):
    current_frametime = 0

    clicked = False
    showed = False

    # frame_data = pd.DataFrame({"LED": [], "Monitor": []})
    led_frames = []
    monitor_frames = []

    while True:
        ret, frame = video.read()
        print(f"current frame: {current_frametime}f")

        if ret:
            show_image_with_size(f"original", frame, show_image_size)

            monitor_mask, monitor_mask_percentage = get_monitor_mask_percentage(frame)
            led_mask, led_mask_percentage = get_led_mask_percentage(frame)

            print("led_mask_percentage", f"{round(led_mask_percentage, 2)}%")
            print("monitor_mask_percentage", f"{round(monitor_mask_percentage, 2)}%")

            print("showed", showed)
            print("clicked", clicked)

            if monitor_mask_percentage >= monitor_mask_on_threshhold and not showed:
                print("Monitor on")
                showed = True

                try:
                    monitor_frames.append(current_frametime)

                except:
                    print("Creating File didn't Work")
            else:
                print("Monitor off")

            if led_mask_percentage >= led_mask_on_threshhold and not clicked:
                print("Click LED on")
                clicked = True
                try:
                    led_frames.append(current_frametime)
                except:
                    print("Creating Files didn't Work")
            else:
                print("Click LED off")

            if led_mask_percentage <= led_mask_off_threshhold and clicked:
                clicked = False

            if monitor_mask_percentage < monitor_mask_off_threshhold and showed:
                showed = False

            show_image_with_size(f"monitor_mask", monitor_mask, show_image_size)
            cv2.imshow(
                f"led_mask",
                led_mask,
            )

            cv2.waitKey(0 if step_by_step else 1)
            current_frametime += 1
            print("-----------------------------------------")
        else:
            create_csv_from_df(video_name, led_frames, monitor_frames)
            break


check_if_path_exists_or_create(export_path)

for video_path in videos:
    print("Processing Video to Data", f"File: {video_path}")
    video_name = Path(video_path).stem
    video = cv2.VideoCapture(video_path)
    process_video(video, video_name)
    video.release()

cv2.destroyAllWindows()

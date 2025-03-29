import os
import gdown
import cv2
import numpy as np
import pickle
import time

def download_missing_video_files(video_link=None, video_name_routine=None):
    if not os.path.exists(video_name_routine):
        print(f"Downloading {video_name_routine}...")
        gdown.download(video_link, video_name_routine, quiet=False)
        if not os.path.exists(video_name_routine):
            raise ValueError(f"Failed to download {video_name_routine}. Check the link.")
    else:
        print(f"{video_name_routine} already exists.")

def run_main_anomaly_loop(video_name_routine=None):
    if video_name_routine is None:
        raise ValueError("No video file provided.")

    cap = cv2.VideoCapture(video_name_routine)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_name_routine}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    anomaly_video_file = 'anomaly_detection.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(anomaly_video_file, fourcc, fps, (frame_width, frame_height))

    fgbg = cv2.createBackgroundSubtractorMOG2()

    with open('routine_map.pkl', 'rb') as f:
        routine_map = pickle.load(f)

    max_val = np.max(routine_map)
    probability_map = routine_map / max_val if max_val != 0 else np.zeros_like(routine_map)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray_frame)
        _, motion_map = cv2.threshold(fgmask, 126, 1, cv2.THRESH_BINARY)

        prediction_map = motion_map * probability_map
        prediction_map[prediction_map == 0] = 1e-6  # Prevent log(0) by substituting a small value

        log_likelihood_map = -np.log(prediction_map)
        log_likelihood_map[log_likelihood_map < 5] = 0
        log_likelihood_map = cv2.morphologyEx(log_likelihood_map, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        output_frame = (log_likelihood_map > 0).astype(np.uint8) * 255
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        out.write(output_frame_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray_frame)
        _, motion_map = cv2.threshold(fgmask, 126, 1, cv2.THRESH_BINARY)
        routine_map += motion_map

        motion_map_rgb = cv2.cvtColor((255 * motion_map).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(motion_map_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    with open('routine_map.pkl', 'wb') as f:
        pickle.dump(routine_map, f)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_main_routine_loop(video_name_routine=None):
    if video_name_routine is None:
        raise ValueError("No video file provided.")

    cap = cv2.VideoCapture(video_name_routine)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_name_routine}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    anomaly_video_file = 'anomaly_detection.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(anomaly_video_file, fourcc, fps, (frame_width, frame_height))

    fgbg = cv2.createBackgroundSubtractorMOG2()

    with open('routine_map.pkl', 'rb') as f:
        routine_map = pickle.load(f)

    max_val = np.max(routine_map)
    probability_map = routine_map / max_val if max_val != 0 else np.zeros_like(routine_map)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray_frame)
        _, motion_map = cv2.threshold(fgmask, 126, 1, cv2.THRESH_BINARY)

        prediction_map = motion_map * probability_map
        prediction_map[prediction_map == 0] = 1

        log_likelihood_map = -np.log(prediction_map)
        log_likelihood_map[log_likelihood_map < 5] = 0
        log_likelihood_map = cv2.morphologyEx(log_likelihood_map, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        output_frame = (log_likelihood_map > 0).astype(np.uint8) * 255
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        out.write(output_frame_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    current_directory = os.getcwd()

    video_name_routine = os.path.join(current_directory, 'routine_frame.mp4')
    download_missing_video_files(
        video_name_routine=video_name_routine,
        video_link='https://drive.google.com/uc?id=1X6QN3wLnglkqTOL-bbVVWvq1RKsTIXY0'
    )
    run_main_routine_loop(video_name_routine)

    video_name_test = os.path.join(current_directory, 'test1.mp4')
    download_missing_video_files(
        video_name_routine=video_name_test,
        video_link='https://drive.google.com/uc?id=18sjBcZ788ZTfVaek_4gIWJRxCYkfFbYF'
    )
    run_main_anomaly_loop(video_name_test)

if __name__ == '__main__':
    main()

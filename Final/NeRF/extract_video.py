import os
import cv2


def extract_images(video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)
    frame_interval = int(5)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # print(frame_interval)
            if count % frame_interval == 0:
                image_name = os.path.join(
                    output_path, f"{count//frame_interval}.jpg")
                cv2.imwrite(image_name, frame)
            count += 1
        else:
            break
    cap.release()


if __name__ == '__main__':
    video_path = 'kipling.mp4'
    output_folder = 'kipling'
    extract_images(video_path, output_folder)
    print("Finished!")

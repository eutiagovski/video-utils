import datetime
import glob
import cv2
import os


def save_image(frame):
    cv2.imwrite(
        f'{str(datetime.datetime.now()).replace(" ", "_")}.jpg', frame)
    print(f'[INFO] Imagem salva na galeria.')


def make_video(clear_images=True):
    images = sorted(glob.glob('*.jpg'))  # Loading the captured image.
    print("Total number of images{0}".format(len(images)))

    if len(images) < 30:  # FPS settings
        frame_rate = 2
    else:
        frame_rate = len(images)/30

    width = 640
    height = 480
    # Specify the video codec as mp. Decide the extension of the video (although it is a little different),
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Specify the information of the video to be created (file name, extension, FPS, video size).
    video = cv2.VideoWriter('{0}.mp4'.format(str(datetime.datetime.now()).replace(
        " ", "_")), fourcc, frame_rate, (width, height))

    print("During video conversion...")

    for i in range(len(images)):
        # Load image
        img = cv2.imread(images[i])
        # Match the size of the image.
        img = cv2.resize(img, (width, height))
        video.write(img)

    video.release()
    print("Video conversion completed")

    if clear_images:
        '''
        Remove stored timelapse images
        '''
        for file in images:
            os.remove(file)

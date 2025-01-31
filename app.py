from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import moviepy.editor as mpy
from moviepy.video.compositing.transitions import slide_in
from moviepy.video.fx import all
from moviepy.editor import *
import glob
import random
from PIL import Image
import cv2
import os
import uuid
import shutil
from sys import argv
app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload/'
VIDEO_FOLDER = 'static/videos/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs('static/vids', exist_ok=True)
def add_title_image(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "temp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/port-hole.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("static/music/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = f'static/videos/final_output3{uid}.mp4'
    # Export the final video with the background music
    composite_clip.write_videofile(output_path)
    mp4_file =  f"static/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file) 
    temp_vid="static/videos/temp.mp4"
    shutil.copyfile(output_path, temp_vid)    
    print(mp4_file)
    VIDEO = output_path
    return VIDEO


# Route to serve uploaded images
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({'filepath': filepath})

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_path = data['image_path']
    points = data['points']
    zoom_level = float(data['zoom'])  # Convert zoom_level to float
    
    output_video = os.path.join(VIDEO_FOLDER, 'zoom_animation.mp4')
    create_zoom_video(image_path, points, zoom_level, output_video)
    return jsonify({'video_path': output_video})

# Function to create zoom effect on a single point

def create_zoom_video(image_path, points, zoom_level, output_video):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    frames = []
    num_frames = 500
    
    # Only using the first point
    point = points[0]
    cx, cy = int(point[0] * w), int(point[1] * h)
    
    # Define the sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5,-1],
                                  [0, -1, 0]])

    for i in range(num_frames):
        #alpha = i / num_frames
        alpha = (i / num_frames) ** 2  # This makes the zoom slower at the start and faster at the end

        zoom_factor = 1 + alpha * zoom_level
        
        # Define the cropping region
        x1 = max(cx - int(w / (2 * zoom_factor)), 0)
        y1 = max(cy - int(h / (2 * zoom_factor)), 0)
        x2 = min(cx + int(w / (2 * zoom_factor)), w)
        y2 = min(cy + int(h / (2 * zoom_factor)), h)
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h))
        
        # Apply sharpening to every 5th frame
        if i % 5 == 0:  # Sharpen every 5th frame (adjust as needed)
            resized = cv2.filter2D(resized, -1, sharpening_kernel)
        
        frames.append(resized)
    
    # Save as video
    video = mpy.ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)
    video.write_videofile(output_video, codec='libx264')
    add_title_image(video_path=output_video)


@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

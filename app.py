from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import uuid
from PIL import Image
from moviepy.editor import (
    VideoFileClip, ImageClip, ColorClip, CompositeVideoClip,
    AudioFileClip, ImageSequenceClip
)
from moviepy.editor import ImageSequenceClip

# Then in the function, use ImageSequenceClip directly
video = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload/'
VIDEO_FOLDER = 'static/videos/'
VIDS_FOLDER = 'static/vids/'
ASSETS_FOLDER = 'static/assets/'
MUSIC_FOLDER = 'static/music/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(VIDS_FOLDER, exist_ok=True)


def add_title_image(video_path):
    """Adds a title image overlay and a background color to the video."""
    hex_colors = ["#A52A2A", "#ad1f1f", "#16765c", "#7a4111", "#9b1050", "#8e215d", "#2656ca"]
    hex_color = np.random.choice(hex_colors)

    video_clip = VideoFileClip(video_path)
    width, height = video_clip.size
    padded_size = (width + 50, height + 50)
    
    # Convert HEX to RGB
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    
    # Background and overlay image
    background = ColorClip(padded_size, color=(r, g, b))
    title_image = ImageClip(f"{ASSETS_FOLDER}port-hole.png").resize(padded_size)
    
    # Create the final composite video
    padded_video = CompositeVideoClip([background, video_clip.set_position("center")])
    composite_clip = CompositeVideoClip([padded_video, title_image.set_position((0, -5))])
    
    # Add background music
    music_files = [f for f in os.listdir(MUSIC_FOLDER) if f.endswith('.mp3')]
    if music_files:
        music_clip = AudioFileClip(os.path.join(MUSIC_FOLDER, np.random.choice(music_files)))
        music_clip = music_clip.set_duration(video_clip.duration).audio_fadein(1).audio_fadeout(1)
        composite_clip = composite_clip.set_audio(music_clip)
    
    output_path = f"{VIDEO_FOLDER}final_output_{uuid.uuid4().hex}.mp4"
    composite_clip.write_videofile(output_path, codec='libx264')

    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handles image upload."""
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({'filepath': filepath})


@app.route('/process', methods=['POST'])
def process():
    """Processes the uploaded image and applies zoom effect."""
    data = request.json
    image_path = data['image_path']
    points = data['points']
    zoom_level = float(data['zoom'])

    output_video = os.path.join(VIDEO_FOLDER, 'zoom_animation.mp4')
    create_zoom_video(image_path, points, zoom_level, output_video)
    return jsonify({'video_path': output_video})


def create_zoom_video(image_path, points, zoom_level, output_video):
    """Applies zoom-in effect on the selected point in the image."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    cx, cy = int(points[0][0] * w), int(points[0][1] * h)
    
    frames = []
    num_frames = 500
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    for i in range(num_frames):
        alpha = (i / num_frames) ** 2
        zoom_factor = 1 + alpha * zoom_level

        x1, y1 = max(cx - w // (2 * zoom_factor), 0), max(cy - h // (2 * zoom_factor), 0)
        x2, y2 = min(cx + w // (2 * zoom_factor), w), min(cy + h // (2 * zoom_factor), h)
        
        cropped = img[int(y1):int(y2), int(x1):int(x2)]
        resized = cv2.resize(cropped, (w, h))
        
        if i % 5 == 0:
            resized = cv2.filter2D(resized, -1, sharpening_kernel)

        frames.append(resized)

    video = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)
    video.write_videofile(output_video, codec='libx264')
    add_title_image(output_video)


@app.route('/video/<filename>')
def video(filename):
    """Serves processed video files."""
    return send_from_directory(VIDEO_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

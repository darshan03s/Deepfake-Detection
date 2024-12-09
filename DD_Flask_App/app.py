import os, shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from utils import *

sequence_length = 120

use_faces = True

path_to_model = "../Models/DD_GRU_Torch_Model_epoch(8)_acc(93.16610705115544)_loss(0.2616659907878248)_testacc(88.61660079051383)_testloss(0.3346584242670173).pt"

UPLOAD_FOLDER = './user_upload'
PREDICTIONS_FOLDER = './predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # Remove files or symbolic links
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove subdirectories

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    empty_folder(UPLOAD_FOLDER)
    empty_folder(PREDICTIONS_FOLDER)
    
    if 'videoFile' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files['videoFile']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})

    if file:
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > 100 * 1024 * 1024:  # 100MB
            return jsonify({"status": "error", "message": "File too large (max: 100MB)"})

        # Save file with the new name
        file_ext = os.path.splitext(file.filename)[1]
        save_path = os.path.join(UPLOAD_FOLDER, f"user_upload{file_ext}")
        file.save(save_path)
        video = [save_path]
        video_dataset = get_dataset(
            video, sequence_length=sequence_length, use_faces=use_faces, model=model)
        print(f"Prediction started")
        prediction, confidence, output = predict_video(model, video_dataset)
        print(
            f"Prediction : {prediction}, Output : {output}, Confidence : {confidence}")
        print(f"Prediction ended")

        return jsonify({"status": "success", "prediction": prediction, "output": output, "confidence": confidence})

    return jsonify({"status": "error", "message": "File upload failed"})

@app.route('/predictions/<filename>')
def serve_prediction(filename):
    return send_from_directory('predictions', filename)

@app.route('/fbf')
def fbf():
    image_files = [f for f in sorted(os.listdir('predictions')) if f.endswith(('.jpg', '.png'))]
    
    return render_template('fbf.html', images=image_files)

@app.route('/img-df')
def imgdf():
    return render_template('img-df.html')

@app.route('/analyse-img', methods=['POST'])
def analyse_img():
    if 'imageFile' not in request.files:
        return "No file part"
    
    file = request.files['imageFile']
    if file.filename == '':
        return "No selected file"
    
    # Save the image to the specified path
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_img.jpg')
    file.save(image_path)
    
    result = predict_img_df(image_path)
    
    return jsonify({"status" : "success", "result" : result})


model = get_model(path_to_model)
print(f"Loaded model")
model.eval()

if __name__ == '__main__':
    app.run(debug=True)

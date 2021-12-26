from flask import Flask, flash, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os

ALLOWED_EXTENSIONS = set(['jpg', 'dicom', 'png'])

secret_key = os.urandom(12)

DETECT_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = DETECT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/best_70.pt')

def detect_image(img_path):
    
    img = cv2.imread("./static/picture/"+img_path)
    # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    colors = [[252, 186, 3], [128, 217, 147], [227, 35, 14], [255, 123, 0], [0, 255, 251], [0, 255, 98], [3, 36, 255], [136, 0, 255], [255, 0, 225], [255, 0, 60], [250, 187, 218], [191, 230, 129], [246, 255, 122], [136, 219, 196]]
    height, width, channels = img.shape

    results = model(img)

    results.print()
    bbox = results.pandas().xyxy[0]

    boxes = []
    confidence = []
    class_name = []
    object_detected = []
    class_id = []

    print(bbox)

    for index, row in bbox.iterrows():
        if row['confidence'] > 0.3:
            print(row['confidence'])

            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence.append(float(row['confidence']))
            class_name.append(row['name'])
            object_detected.append([row['name'], int(row['confidence']*100)])
            class_id.append(row['class'])

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        label = str(class_name[i])  + " - " + str(int(confidence[i]*100))+"%"
        color = colors[class_id[i]]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
        cv2.putText(img, label, (int(xmin), int(ymin)), font, 1, color, 2)

    cv2.imwrite("./static/detect/"+img_path, img)
    return object_detected

def save_img_from_dcm(img_dir, image_id, voi_lut=True, fix_monochrome=True):
    img_fp = os.path.join(img_dir, "{}.jpg".format(image_id))
    if os.path.exists(img_fp):
        return
    dcm_fp = os.path.join(img_dir, "{}.dicom".format(image_id))
    print(dcm_fp)
    dicom = pydicom.read_file(dcm_fp)

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    img_fp = os.path.join(img_dir, "{}.jpg".format(image_id))
    cv2.imwrite(img_fp, data)

def image_clahe(filename):
    image = cv2.imread(filename, 1)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, image)
    cv2.imwrite(filename, image)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            split_filename = filename.split(".")
            save_folder = os.path.join('static', 'picture')
            file.save(os.path.join(save_folder, filename))
            flash('file successfully uploaded')
            img_path = os.path.join(save_folder, filename)
            if split_filename[1] == "dicom":
                save_img_from_dcm(save_folder, split_filename[0])
                filename = split_filename[0] + ".jpg"
            image_path = os.path.join(save_folder, "{}".format(filename))
            # image_clahe(image_path)
            return redirect(url_for('result', filename=split_filename[0]))
        else:
            flash('Allowed file types are jpg and dicom')
            return redirect(request.url)


@app.route("/result/<filename>")
def result(filename):
    image_path = "./static/picture"
    extension = "jpg"
    file_path = os.path.join(image_path, "{}.png".format(filename))
    if os.path.exists(file_path):
        extension = "png"
    img_path = filename + "." + extension
    object_detected = detect_image(img_path)
    return render_template("result.html", original_image=img_path, detected_image=img_path, object_detected=object_detected)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
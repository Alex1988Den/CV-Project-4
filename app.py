from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path, max_dim=512):
    img = Image.open(image_path).convert("RGB")
    long = max(img.size)
    scale = max_dim / long
    new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    return tf.convert_to_tensor(img, dtype=tf.float32)

def get_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    return tf.keras.Model([vgg.input], outputs)

def compute_loss(model, target_image, content_image, style_image):
    target_outputs = model(target_image)
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    
    content_loss = tf.reduce_mean((content_outputs[0] - target_outputs[0])**2)
    
    style_loss = 0
    for a, b in zip(style_outputs[1:], target_outputs[1:]):
        gram_a = tf.linalg.einsum('bijc,bijd->bcd', a, a)
        gram_b = tf.linalg.einsum('bijc,bijd->bcd', b, b)
        style_loss += tf.reduce_mean((gram_a - gram_b)**2)
    
    loss = content_loss + 1e-4 * style_loss
    return loss

@tf.function
def train_step(model, target_image, content_image, style_image, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, target_image, content_image, style_image)
    grad = tape.gradient(loss, target_image)
    optimizer.apply_gradients([(grad, target_image)])
    target_image.assign(tf.clip_by_value(target_image, 0.0, 1.0))
    return loss

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return "Ошибка: загрузите и контентное, и стилевое изображение.", 400

    content_image = request.files['content_image']
    style_image = request.files['style_image']

    if content_image and style_image and allowed_file(content_image.filename) and allowed_file(style_image.filename):
        content_filename = secure_filename(content_image.filename)
        style_filename = secure_filename(style_image.filename)

        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

        content_image.save(content_image_path)
        style_image.save(style_image_path)

        try:
            content_img = load_image(content_image_path)
            style_img = load_image(style_image_path)

            vgg_model = get_vgg_model()

            target_image = tf.Variable(content_img[tf.newaxis, ...], dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

            epochs = 100
            for epoch in range(epochs):
                loss = train_step(vgg_model, target_image, content_img[tf.newaxis, ...], style_img[tf.newaxis, ...], optimizer)

            output_image = target_image.numpy()[0]
            output_image = np.clip(output_image, 0.0, 1.0)
            output_image = Image.fromarray((output_image * 255).astype(np.uint8))
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_stylized_image.jpg')
            output_image.save(output_path)

            return render_template('transfer.html', output_image_path=output_path)

        except Exception as e:
            return f"Ошибка: {e}", 500

    return "Ошибка: недопустимый формат файлов.", 400

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000)  # Использует порт, предоставленный Render
    app.run(host="0.0.0.0", port=port)  # Слушаем на всех интерфейсах

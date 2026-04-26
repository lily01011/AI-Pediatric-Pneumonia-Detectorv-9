import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    'densenet121_best_model.keras',
    compile=False,
    safe_mode=False
)
print("✅ Model loaded!")

# GradCAM layer
last_conv_layer = 'conv5_block16_concat'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    base_model = model.get_layer('densenet121')
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, base_predictions = grad_model(inputs)
        class_channel = base_predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    final_pred = model(img_array, training=False)
    pred_score = float(final_pred.numpy()[0][0])
    return heatmap, pred_score


def run_gradcam(img_path, true_label):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array / 255.0, axis=0)

    heatmap, pred_score = make_gradcam_heatmap(
        img_array_expanded, model, last_conv_layer
    )

    pred_label = "PNEUMONIA" if pred_score > 0.260 else "NORMAL"
    confidence = pred_score if pred_score > 0.5 else 1 - pred_score

    original = np.uint8(img_array)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    correct = "CORRECT" if pred_label == true_label else "WRONG"
    fig.suptitle(
        f'Grad-CAM | True: {true_label} | Predicted: {pred_label} '
        f'({confidence*100:.1f}%) | {correct}',
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(original)
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Heatmap (AI Focus Area)')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'gradcam_result_{true_label}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Predicted: {pred_label} | Confidence: {confidence*100:.1f}%")


# --- TEST IT --- put any chest xray image path here
run_gradcam('your_test_image.jpg', 'PNEUMONIA')
import gradio as gr
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import uuid
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from inference_bot import run_pipeline  

import matplotlib.pyplot as plt 

dark_css = """
body { background-color: #111; color: #eee; font-family: Arial, sans-serif; }
.gr-button { background-color: #4CAF50; color: #fff; font-weight: bold; }
#advice_box textarea {
    background-color: #1f1f1f !important;
    color: #f0f0f0 !important;
    font-family: 'Consolas', monospace;
    font-size: 14px;
    border-radius: 10px;
    padding: 12px;
}
#disease_name textarea {
    font-size: 22px !important;
    color: #FF9800 !important;
    font-weight: bold;
    text-align: center;
}
.gr-row { margin-top: 15px; }
"""

TRAIN_DIR = r"C:\\Users\\DELL\\Desktop\\projects\\grape disease detection\\data\\train"
class_labels = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])


def predict_and_advice(image_pil):
    """Run prediction and generate LLM advice"""
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_pil.save(temp_filename)
    
    predicted_class, confidence, advice = run_pipeline(temp_filename)
    os.remove(temp_filename)
    
    return predicted_class, f"{advice}\n\nConfidence: {confidence*100:.2f}%", image_pil


def generate_pdf(image_pil, disease_name, advice_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 50, "🍇 Grape Leaf Disease Report")
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 100, f"Disease Detected: {disease_name}")

    img_width, img_height = image_pil.size
    aspect = img_height / img_width
    display_width = 4*inch
    display_height = display_width * aspect
    image_stream = BytesIO()
    image_pil.save(image_stream, format="PNG")
    image_stream.seek(0)
    c.drawImage(ImageReader(image_stream), 50, height - 100 - display_height - 20, width=display_width, height=display_height)

    c.setFont("Helvetica", 12)
    text_object = c.beginText(50, height - 100 - display_height - 60)
    text_object.setLeading(18)
    
    from reportlab.lib.utils import simpleSplit

    max_width = width - 100 
    font_name = "Helvetica"
    font_size = 12

    for line in advice_text.split("\n"):
        wrapped_lines = simpleSplit(line, font_name, font_size, max_width)
        for wrapped_line in wrapped_lines:
            text_object.textLine(wrapped_line)
    
    c.drawText(text_object)
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer


def generate_confidence_graph(pred_class):
    import tensorflow as tf
    from inference_bot import model  


    return None


def predict_and_generate_pdf(img):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    img.save(temp_filename)

    from inference_bot import model
    img_resized = Image.open(temp_filename).convert("RGB").resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    os.remove(temp_filename)

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    pred_class = class_labels[pred_idx]

    _, advice, img_out = predict_and_advice(img)
    pdf_buffer = generate_pdf(img_out, pred_class, advice)
    temp_pdf = f"report_{uuid.uuid4().hex}.pdf"
    with open(temp_pdf, "wb") as f:
        f.write(pdf_buffer.getbuffer())

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(class_labels, preds, color="#4CAF50", edgecolor="white")
    ax.set_title("Model Confidence per Disease Class", color="white", fontsize=12, pad=10)
    ax.set_ylabel("Confidence", color="white")
    ax.set_xlabel("Disease Classes", color="white")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height()*100:.1f}%", ha="center", color="white", fontsize=10)

    plt.tight_layout()

    return pred_class, advice, img_out, temp_pdf, fig 


with gr.Blocks(css=dark_css) as demo:
    gr.Markdown("<h1 style='text-align:center; color:#4CAF50'>🍇 Grape Leaf Disease Detector</h1>")
    gr.Markdown("<p style='text-align:center; color:#f0f0f0'>Upload a grape leaf image to detect the disease and get professional recommendations from an AI expert.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Leaf Image", type="pil", image_mode="RGB")
            predict_button = gr.Button("Detect Disease & Get Advice")
        with gr.Column(scale=2):
            disease_name = gr.Textbox(label="Detected Disease", interactive=False, elem_id="disease_name", lines=1)
            output_text = gr.Textbox(label="Precautions & Recommendations", lines=20, interactive=False, placeholder="Prediction and advice will appear here...", elem_id="advice_box")
            output_image = gr.Image(label="Uploaded Image", interactive=False)
            confidence_plot = gr.Plot(label="Model Confidence Graph")
            download_pdf = gr.File(label="Download PDF Report", file_types=[".pdf"])

    predict_button.click(
        fn=predict_and_generate_pdf,
        inputs=image_input,
        outputs=[disease_name, output_text, output_image, download_pdf, confidence_plot] 
    )

demo.launch(share=True)

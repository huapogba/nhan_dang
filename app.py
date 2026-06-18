"""
Medical Image Classification Demo
Single Model Inference
Supports: Brain Tumor, Chest X-Ray, HAM10000
Run: python app.py
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import gradio as gr

from PIL import Image

# =============================================================================
# IMPORT YOUR SPIKING RESFORMER MODEL
# =============================================================================
from spikingjelly.activation_based import surrogate, neuron
from SpikResformer import spikingresformer_ti
from SEW_ResNet import sew_resnet18
from SpikFormer import spikformer

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# MODEL_PATHS = {
#     "Brain Tumor": ["weightload/best_model_Resformer_ti_brain_tumour.pth",
#                     "best_model_Spikformer_brain_tumor_new.pth",
#                     "best_model_SEW18_brain_tumor.pth"
#     ],
#     "Chest X-Ray": "best_model_Resformer_chest_xray.pth",
#     "HAM10000": "best_model_Resformer_OG (7).pth",
# }
MODEL_CONFIGS = {
    "Brain Tumor": {
        "Spiking Resformer": {
            "path": "weightload/best_model_Resformer_ti_brain_tumour.pth",
            "type": "resformer"
        },
        "SpikFormer": {
            "path": "best_model_Spikformer_brain_tumor_new.pth",
            "type": "spikformer"
        },
        "SEW ResNet18": {
            "path": "best_model_SEW18_brain_tumor.pth",
            "type": "sew"
        }
    }
}
DATASET_LABELS = {

    "Brain Tumor": [
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary"
    ],

    "Chest X-Ray": [
        "normal",
        "pneumonia",
    ],

    "HAM10000": [
        "akiec",
        "bcc",
        "bkl",
        "df",
        "mel",
        "nv",
        "vasc"
    ],
}

DATASET_COLORS = {
    "Brain Tumor": "#3784d9",
    "Chest X-Ray": "#1d9e75",
    "HAM10000": "#d85a30",
}

IMG_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Cache loaded models
LOADED_MODELS = {}


# ─── IMAGE TRANSFORM ─────────────────────────────────────────────────────────

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─── EXTRACT LABELS FROM CHECKPOINT ──────────────────────────────────────────

def try_extract_labels(ckpt):

    keys = [
        "class_names",
        "classes",
        "idx_to_class",
        "label_names",
        "class_to_idx",
        "labels"
    ]

    for key in keys:

        if key not in ckpt:
            continue

        val = ckpt[key]

        if isinstance(val, list):
            return [str(v) for v in val]

        if isinstance(val, dict):

            # idx -> class
            if all(isinstance(k, int) for k in val.keys()):
                return [val[i] for i in sorted(val.keys())]

            # class -> idx
            inv = {v: k for k, v in val.items()}
            return [inv[i] for i in sorted(inv.keys())]

    return None


# ─── BUILD MODEL ─────────────────────────────────────────────────────────────

def build_model(model_type,num_classes):

    if model_type == "resformer":
        return spikingresformer_ti(
            num_classes=num_classes
        )

    elif model_type == "spikformer":
        return spikformer(
            embed_dims=256,
            num_heads=4,
            depths=4,
            num_classes=4)

    elif model_type == "sew":
        model = sew_resnet18(
            pretrained=False,
            cnf='ADD',
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan()
        )
        model.fc = nn.Linear(512, num_classes)
        return model

    raise ValueError(f"Unknown model type: {model_type}")


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────

def load_model(dataset, model_name):

    cache_key = f"{dataset}_{model_name}"

    if cache_key in LOADED_MODELS:
        return LOADED_MODELS[cache_key]

    cfg = MODEL_CONFIGS[dataset][model_name]

    path = cfg["path"]
    model_type = cfg["type"]

    if not os.path.isfile(path):

        print(f"[ERROR] File not found: {path}")
        return None

    try:

        ckpt = torch.load(
            path,
            map_location=DEVICE
        )

        if isinstance(ckpt, dict):

            if "model" in ckpt:
                state_dict = ckpt["model"]

            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]

            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]

            else:
                state_dict = ckpt

        else:
            state_dict = ckpt

        labels = DATASET_LABELS[dataset]
        num_classes = len(labels)

        model = build_model(
            model_type,
            num_classes
        )

        clean_state_dict = {}

        for k, v in state_dict.items():
            clean_state_dict[k.replace("module.", "")] = v

        missing, unexpected = model.load_state_dict(
            clean_state_dict,
            strict=False
        )

        if missing:
            print("Missing:", missing)

        if unexpected:
            print("Unexpected:", unexpected)

        model.to(DEVICE)
        model.eval()

        bundle = {
            "model": model,
            "labels": labels
        }

        LOADED_MODELS[cache_key] = bundle

        print(f"[INFO] Loaded {model_name}")
        return bundle

    except Exception as e:

        print(f"[ERROR] Failed loading {path}")
        print(e)

        return None

# ─── INFERENCE ───────────────────────────────────────────────────────────────

def predict(image, dataset, model_name):

    if image is None:

        return """
        <p style='color:#9ca3af;padding:8px;'>
        Upload an image first.
        </p>
        """

    bundle = load_model(dataset,model_name)

    if bundle is None:

        return """
        <p style='color:red;padding:8px;'>
        Failed loading model.
        </p>
        """

    tensor = TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = bundle["model"](tensor)

        probs = torch.softmax(
            logits,
            dim=1
        ).squeeze().cpu().numpy()

    labels = bundle["labels"]

    preds = {
        labels[i]: float(probs[i]) * 100
        for i in range(len(labels))
    }

    preds = dict(
        sorted(
            preds.items(),
            key=lambda x: -x[1]
        )
    )

    color = DATASET_COLORS[dataset]

    top_label = list(preds.keys())[0]
    top_conf = preds[top_label]

    # -------------------------------------------------------------
    # HTML OUTPUT
    # -------------------------------------------------------------

    html = f"""
    <div style='padding:8px 0;'>

    <div style='border:1px solid #e5e7eb;
                border-radius:14px;
                padding:20px;'>

    <div style='display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:18px;'>

    <div>

    <div style='font-size:13px;
                color:#6b7280;
                margin-bottom:3px;'>

    {dataset}

    </div>

    <div style='font-size:22px;
                font-weight:600;
                color:#111;'>

    {top_label}

    </div>

    </div>

    <div style='background:{color}20;
                color:{color};
                padding:8px 14px;
                border-radius:999px;
                font-weight:600;
                font-size:14px;'>

    {top_conf:.2f}%

    </div>

    </div>
    """

    for label, conf in preds.items():

        html += f"""
        <div style='margin-bottom:12px;'>

        <div style='display:flex;
                    justify-content:space-between;
                    font-size:14px;
                    margin-bottom:4px;'>

        <span>{label}</span>
        <span>{conf:.2f}%</span>

        </div>

        <div style='background:#f3f4f6;
                    height:7px;
                    border-radius:999px;'>

        <div style='width:{conf}%;
                    background:{color};
                    height:7px;
                    border-radius:999px;'>
        </div>

        </div>

        </div>
        """

    html += "</div></div>"

    return html


# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
}

footer {
    display: none !important;
}
"""


# ─── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(
    css=CSS,
    title="Medical Image Classification"
) as demo:

    gr.HTML("""
    <div style='padding:20px 0 10px 0;'>

    <h1 style='font-size:28px;
               font-weight:600;
               margin-bottom:6px;'>

    Medical Image Classification

    </h1>

    <p style='color:#6b7280;
              font-size:15px;'>

    Spiking Resformer inference demo

    </p>

    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Input Image",
                height=300
            )

            dataset_sel = gr.Dropdown(
                choices=list(DATASET_LABELS.keys()),
                value="Brain Tumor",
                label="Dataset"
            )
            model_sel = gr.Dropdown(
                choices=[
                    "Spiking Resformer",
                    "SpikFormer",
                    "SEW ResNet18"
                ],
                value="Spiking Resformer",
                label="Model"
            )

            run_btn = gr.Button(
                "Run Inference",
                variant="primary"
            )

        with gr.Column(scale=1):

            output_html = gr.HTML("""
            <p style='color:#9ca3af;padding-top:20px;'>
            Upload image and run inference.
            </p>
            """)

    gr.HTML(f"""
    <div style='margin-top:16px;
                font-size:12px;
                color:#9ca3af;'>

    Running on: <strong>{DEVICE}</strong>

    </div>
    """)

    run_btn.click(
        fn=predict,
        inputs=[
            image_input,
            dataset_sel,
            model_sel
        ],
        outputs=output_html
    )


# ─── START ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    demo.launch(
        share=False
    )
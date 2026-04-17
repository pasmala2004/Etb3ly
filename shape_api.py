from flask import Flask, request, jsonify, send_file
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import torch
import base64
import uuid
import os
from PIL import Image
import io

app = Flask(__name__)

print("Loading model...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini',
    torch_dtype=torch.float16
)
pipeline.to('cuda')
print("Model ready!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        img_data = None
        content_type = request.content_type or ''

        # ── Handle JSON ──────────────────────────────
        if 'application/json' in content_type:
            data = request.json
            if not data or 'image' not in data:
                return jsonify({"error": "Missing 'image' field in JSON body"}), 400
            img_data = base64.b64decode(data['image'])

        # ── Handle multipart/form-data ───────────────
        elif 'multipart/form-data' in content_type:
            file = request.files.get('image')
            if not file:
                return jsonify({"error": "Missing 'image' field in form data"}), 400
            img_data = file.read()

        # ── Handle raw binary ────────────────────────
        elif 'image/' in content_type:
            img_data = request.data

        # ── Unsupported ──────────────────────────────
        else:
            return jsonify({
                "error": f"Unsupported Content-Type: {content_type}",
                "supported": [
                    "application/json  → body: {image: base64string}",
                    "multipart/form-data → field: image (file)",
                    "image/jpeg or image/png → raw binary"
                ]
            }), 415

        # ── Process Image ────────────────────────────
        image = Image.open(io.BytesIO(img_data)).convert('RGB')

        temp_id = str(uuid.uuid4())
        input_path = f'/tmp/{temp_id}.png'
        output_path = f'/tmp/{temp_id}.stl'
        image.save(input_path)

        # ── Generate 3D Shape ────────────────────────
        print(f"Generating shape for job {temp_id}...")
        output = pipeline(image=input_path)
        mesh = output[0]
        mesh.export(output_path)

        # ── Cleanup & Return ─────────────────────────
        os.remove(input_path)
        print(f"Done! Job {temp_id} complete")

        return send_file(
            output_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='model.stl'
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
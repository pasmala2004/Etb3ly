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

@app.route('/', methods=['GET'])
def health():
    return jsonify({"statushealth": "running"})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        img_data = None
        
        # 1. Try Multipart Form (Postman 'form-data')
        if 'image' in request.files:
            file = request.files['image']
            img_data = file.read()
            print("Received via multipart/form-data")

        # 2. Try JSON (Postman 'raw' -> JSON)
        elif request.is_json:
            data = request.get_json()
            if 'image' in data:
                img_data = base64.b64decode(data['image'])
                print("Received via JSON")

        # 3. Try Raw Binary (Postman 'binary')
        elif request.data:
            img_data = request.data
            print("Received via raw binary")

        # Fallback if nothing is found
        if not img_data:
            return jsonify({
                "error": "No image data found",
                "message": "Send image as form-data (key: 'image'), JSON (key: 'image'), or raw binary."
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
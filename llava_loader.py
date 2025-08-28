import subprocess
import shlex
import time
import json
import re
from PIL import Image, ImageDraw

def run_llava_stream(
    image_path: str,
    prompt: str,
    model_path: str = "/home/spinoandraptos/Downloads/llava-v1.5-7b-Q4_K_M.gguf",
    mmproj_path: str = "/home/spinoandraptos/Downloads/llava-v1.5-7b-mmproj-f16.gguf",
    cli_path: str = "./mllmApp/android/llama.cpp/build/bin/llama-mtmd-cli"
):
    """
    Run LLaVA inference using the CLI, stream output line by line, and track inference time.
    """
    cmd = f'{cli_path} -m {model_path} --mmproj {mmproj_path} --image {image_path} -p "{prompt}" --chat-template vicuna'
    start_time = time.time()

    # Open subprocess for streaming output
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    print("Inference output:\n")
    for line in process.stdout:
        line = line.strip()
        if line:
            output_lines.append(line)
            print(line)

    process.wait()
    end_time = time.time()
    duration = end_time - start_time

    if process.returncode != 0:
        raise RuntimeError(f"LLaVA exited with code {process.returncode}")

    print(f"\nInference completed in {duration:.2f} seconds.")
    return "\n".join(output_lines)


def extract_json_from_llava_output(llava_output: str):
    """
    Extract the first JSON array from the LLaVA output string.
    Returns a Python list or raises an error if no JSON is found.
    """
    match = re.search(r"\[\s*\{.*?\}\s*\]", llava_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLaVA output.")
    json_str = match.group(0)
    return json.loads(json_str)

def draw_bboxes(image_path: str, objects: list, save_path: str = None):
    """
    Draw bounding boxes on the image.
    `objects` should be a list of dicts with keys: type, bbox
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    colors = {"face": "red", "license_plate": "blue", "document": "green"}

    for obj in objects:
        obj_type = obj.get("type", "unknown")
        bbox = obj.get("bbox", None)
        if bbox and len(bbox) == 4:
            # If bbox is normalized (0–1), scale to image size
            if max(bbox) <= 1.0:
                w, h = image.size
                bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
            draw.rectangle(bbox, outline=colors.get(obj_type, "yellow"), width=3)
            draw.text((bbox[0], bbox[1] - 10), obj_type, fill=colors.get(obj_type, "yellow"))
        else:
            print("Skipping invalid bbox:", obj)

    image.show()
    if save_path:
        image.save(save_path)
        print(f"Image saved to {save_path}")


if __name__ == "__main__":
    image_file = "/home/spinoandraptos/Downloads/test.jpg"

    prompt = """
    You are a sensitive content detector. Analyze the image and identify sensitive regions such as:

    - Human faces
    - License plates
    - Documents (IDs, passports, receipts, forms)
    - Identity numbers (social security numbers, credit card numbers)
    - Personal addresses
    - Phone numbers
    - Email addresses
    - Other personally identifiable information (PII)

    Return the results as a JSON array with objects in this format:

    [
    {
        "type": "face",
        "bbox": [x_min, y_min, x_max, y_max]
    }
    ]

    Only return valid JSON. Do not include any extra explanation.
    """

    try:
        result = run_llava_stream(image_file, prompt)
        objects = extract_json_from_llava_output(result)
        draw_bboxes(
            image_path="/home/spinoandraptos/Downloads/test.jpg",
            objects=objects,
            save_path="/home/spinoandraptos/Downloads/test_bboxes.jpg"
        )
    except Exception as e:
        print("Failed:", e)

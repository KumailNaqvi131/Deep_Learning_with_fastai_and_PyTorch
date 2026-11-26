"""
Chapter 2 — From Model to Production
Topic 6: Turning your model into an online application (book sections "Creating a Notebook App
from the Model", "Turning Your Notebook into a Real App" and "Deploying Your App")

The book builds the app with Jupyter widgets (an upload button, a "Classify" button and a label)
and turns the notebook into a web app with Voilà. The same idea with nothing but Python's
standard library: a tiny web server with an upload form that runs `learn_inf.predict` on the
uploaded image. It uses the model exported by 05_export_and_inference.py (on the CPU: for one
image at a time a CPU is fine, as the book points out).

Run:
    python 06_web_app.py                 # then open http://localhost:8000 and upload a bear picture
    python 06_web_app.py --selftest      # start, send itself one picture, print the answer, stop
"""

import argparse
import html
import io
import sys
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from common import *  # noqa: F401,F403,E402

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--selftest", action="store_true")
args = parser.parse_args()

model_file = outputs_dir(__file__) / "export.pkl"
if not model_file.exists():
    sys.exit("No exported model yet: run  python 05_export_and_inference.py  first.")
learn_inf = load_learner(model_file)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Bear classifier</title></head>
<body style="font-family:sans-serif;max-width:40em;margin:2em auto">
<h1>Bear classifier</h1><p>Upload a picture of a grizzly, black or teddy bear.</p>
<form method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*">
<button type="submit">Classify</button></form><p style="font-size:1.3em">{result}</p></body></html>"""


def parse_upload(body, content_type):
    """Extract the uploaded file's bytes from a multipart/form-data request body."""
    boundary = content_type.split("boundary=")[1].encode()
    for part in body.split(b"--" + boundary):
        if b"filename=" in part:
            return part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, text):
        data = PAGE.format(result=text).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send("")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        data = parse_upload(body, self.headers["Content-Type"])
        try:
            img = Image.open(io.BytesIO(data))
            pred, _, probs = learn_inf.predict(img)
            self._send(f"Prediction: <b>{html.escape(pred)}</b>; Probability: {probs.max():.4f}")
        except Exception as e:
            self._send(f"Could not read that image ({html.escape(type(e).__name__)})")

    def log_message(self, *a):
        pass


server = HTTPServer(("127.0.0.1", args.port), Handler)
if not args.selftest:
    print(f"Serving on http://localhost:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()
else:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    img_file = get_image_files(untar_data(URLs.BEARS) / "black")[0]
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{img_file.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + img_file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    answer = urllib.request.urlopen(req).read().decode()
    print("Uploaded", img_file.name, "->", answer.split("<p style=\"font-size:1.3em\">")[1].split("</p>")[0])
    server.shutdown()

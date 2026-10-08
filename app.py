import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_text
from src.preprocessing import TextPreprocessor
from src.config import METRICS_PATH, IMAGES_DIR, WEB_DIR

PORT = 8000

class HateSpeechRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving static web UI files and REST API endpoints.
    """
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        # REST API Routes
        if parsed_path == "/api/stats":
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    data = json.load(f)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": "Metrics not found. Train model first."}).encode("utf-8"))
            return

        elif parsed_path == "/api/preprocessing_summary":
            preprocessor = TextPreprocessor()
            report = preprocessor.generate_preprocessing_report()
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"report": report}).encode("utf-8"))
            return

        # Static File Serving
        if parsed_path == "/" or parsed_path == "/index.html":
            filepath = os.path.join(WEB_DIR, "index.html")
            content_type = "text/html"
        elif parsed_path.startswith("/web/"):
            filepath = os.path.join(BASE_DIR, parsed_path.lstrip("/"))
            if parsed_path.endswith(".css"):
                content_type = "text/css"
            elif parsed_path.endswith(".js"):
                content_type = "application/javascript"
            else:
                content_type = "text/plain"
        elif parsed_path.startswith("/static/"):
            filepath = os.path.join(BASE_DIR, parsed_path.lstrip("/"))
            if parsed_path.endswith(".png"):
                content_type = "image/png"
            elif parsed_path.endswith(".jpg") or parsed_path.endswith(".jpeg"):
                content_type = "image/jpeg"
            else:
                content_type = "application/octet-stream"
        else:
            filepath = os.path.join(WEB_DIR, parsed_path.lstrip("/"))
            content_type = "text/html"

        if os.path.exists(filepath) and os.path.isfile(filepath):
            self._set_headers(200, content_type)
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._set_headers(404, "text/html")
            self.wfile.write(b"<h1>404 File Not Found</h1>")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        if parsed_path == "/api/predict":
            text = payload.get("text", "")
            result = predict_text(text)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif parsed_path == "/api/batch_predict":
            texts = payload.get("texts", [])
            results = [predict_text(t) for t in texts]
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"results": results}).encode("utf-8"))
            return

        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, HateSpeechRequestHandler)
    print("================================================================================")
    print(f"[SERVER STARTED] HateGuard AI Web App & REST API Server running at http://localhost:{PORT}")
    print("================================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")

if __name__ == "__main__":
    run_server()

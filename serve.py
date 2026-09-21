#!/usr/bin/env python3
"""Dev server with no-cache headers — use instead of `python3 -m http.server`."""
import http.server
import socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} - {fmt % args}")

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 8000), Handler) as httpd:
    print("http://127.0.0.1:8000  (no-cache mode)")
    httpd.serve_forever()

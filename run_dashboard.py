
"""
run_dashboard.py
─────────────────────────────────────────────────────────────────────
Launch script for the Alzheimer's PET XAI Interactive Dashboard.

Steps performed:
  1. Runs dashboard/prepare_assets.py to crop brain slices from the grid.
  2. Finds a free port starting from 8000.
  3. Launches a local Python HTTP server.
  4. Automatically opens the dashboard in your default web browser.
─────────────────────────────────────────────────────────────────────
"""

import os
import sys
import socket
import webbrowser
import http.server
import socketserver
from threading import Timer

# Add dashboard to python path to import asset compiler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard"))

try:
    from prepare_assets import crop_assets
except ImportError:
    print("Error: Could not import prepare_assets from the dashboard folder.")
    sys.exit(1)


def find_free_port(start_port=8000, max_port=8100):
    """Find and return a free local TCP port."""
    port = start_port
    while port <= max_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                port += 1
    raise IOError("Could not find a free port in the range 8000-8100.")


def open_browser(port):
    """Callback to open default browser after server starts."""
    url = f"http://127.0.0.1:{port}/dashboard/index.html"
    print(f"\n[Dashboard] Automatically opening browser: {url}")
    webbrowser.open(url)


def main():
    print("=" * 60)
    print("  LAUNCHING ALZHEIMER'S PET XAI DASHBOARD")
    print("=" * 60)

    # 1. Prepare assets (crop grid cells)
    print("\n[Step 1/3] Preparing visual slice assets...")
    success = crop_assets()
    if not success:
        print("[Warning] Asset cropping encountered issues. The playground may lack images.")
    else:
        print("[Success] Visual slices cropped and saved to dashboard/assets/.")

    # 2. Find a free port
    print("\n[Step 2/3] Finding an available local port...")
    try:
        port = find_free_port(start_port=8000)
        print(f"[Success] Port identified: {port}")
    except Exception as e:
        print(f"[Error] Failed to find free port: {e}")
        sys.exit(1)

    # 3. Start local HTTP server serving from workspace root
    print("\n[Step 3/3] Starting web server...")
    
    # SimpleHTTPRequestHandler serves files relative to current directory
    handler = http.server.SimpleHTTPRequestHandler
    
    # We run webbrowser opening on a short delay to allow the server to start binding
    Timer(1.5, open_browser, args=[port]).start()

    # Serve from project root (so paths like ../outputs/confusion_matrix.png work)
    try:
        with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
            print(f"  -> Dashboard Server running at: http://127.0.0.1:{port}/dashboard/index.html")
            print("  -> Root folder served successfully.")
            print("  -> Press [Ctrl + C] to terminate the server at any time.")
            print("=" * 60)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Server stopped by user request. Exiting.")
    except Exception as e:
        print(f"\n[Error] Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Launcher script to start both FastAPI backend and Vite frontend together."""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def check_python_deps():
    try:
        import fastapi
        import uvicorn
        import pydantic
    except ImportError:
        print("[!] FastAPI dependencies not found in current Python environment.")
        print("[*] Installing web backend dependencies (fastapi, uvicorn, python-multipart)...")
        req_file = BACKEND_DIR / "requirements.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
        print("[+] Backend dependencies installed successfully!\n")


def check_node_deps():
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.is_dir():
        print("[!] Frontend node_modules not found.")
        print("[*] Running 'npm install' in frontend directory...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), shell=True, check=True)
        print("[+] Frontend packages installed successfully!\n")


def main():
    print("=" * 65)
    print(" 🍃 Ceylon Tea Leaf Disease Diagnostic Web Application 🍃")
    print("=" * 65)

    check_python_deps()
    check_node_deps()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    print("\n[*] Starting FastAPI backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "web.backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT), env=env)

    # Give backend a moment to start
    time.sleep(1.5)

    print("[*] Starting Vite React frontend on http://localhost:5173 ...")
    frontend_cmd = ["npm", "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), shell=True)

    time.sleep(2)
    print("\n" + "=" * 65)
    print("  🚀 Web Application is RUNNING!")
    print("  🌐 Frontend URL:  http://localhost:5173")
    print("  🔌 Backend API:   http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C in this terminal to stop both servers.")
    print("=" * 65 + "\n")

    try:
        webbrowser.open("http://localhost:5173")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend terminated unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n[*] Stopping servers...")
    finally:
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
        except Exception:
            pass
        print("[+] Servers stopped. Goodbye!")


if __name__ == "__main__":
    main()

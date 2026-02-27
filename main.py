import subprocess
import time
import sys
import os

def run_backend(env):
    """Start the FastAPI backend server."""
    # We use 'api.main:app' and --app-dir 'src' so Uvicorn finds the 'core' module correctly
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"],
        env=env
    )

def run_frontend(env):
    """Start the Streamlit frontend."""
    # Run streamlit from the src folder so it has correct context
    src_dir = os.path.join(os.getcwd(), "src")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py"],
        env=env,
        cwd=src_dir
    )

def main():
    # Set PYTHONPATH to include the 'src' directory
    env = os.environ.copy()
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src")
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir

    print("\n" + "="*50)
    print("🚀  NEXUS INTEGRATED RUNNER")
    print("="*50)
    
    try:
        # Start Backend
        print("\n📡 Starting Backend (FastAPI)...")
        backend = run_backend(env)
        print("✅ Backend process started (PID: {})".format(backend.pid))
        
        # Wait for backend to initialize
        print("⏳ Waiting for backend to warm up...")
        time.sleep(3)
        
        # Start Frontend
        print("\n🎨 Starting Frontend (Streamlit)...")
        frontend = run_frontend(env)
        print("✅ Frontend process started (PID: {})".format(frontend.pid))
        
        print("\n" + "-"*50)
        print("🔗 API:       http://localhost:8000")
        print("🔗 UI:        http://localhost:8501")
        print("-"*50)
        print("\nPress Ctrl+C to stop both processes.\n")

        # Monitor processes
        while True:
            if backend.poll() is not None:
                print("\n❌ Backend process exited unexpectedly.")
                break
            if frontend.poll() is not None:
                print("\n❌ Frontend process exited unexpectedly.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑  Shutting down NEXUS...")
    finally:
        if 'backend' in locals():
            backend.terminate()
        if 'frontend' in locals():
            frontend.terminate()
        print("👋  Processes terminated. Goodbye!\n")

if __name__ == "__main__":
    main()

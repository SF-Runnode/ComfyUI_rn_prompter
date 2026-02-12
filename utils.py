import sys
import io
import time
import threading
import os
import re
import uuid

if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

PREFIX = "✨"
ERROR_PREFIX = "✨-❌"
PROCESS_PREFIX = "✨"
REQUEST_PREFIX = "✨"
WARN_PREFIX = "✨-⚠️"

def _enable_windows_vt():
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

_enable_windows_vt()

def sanitize_sensitive_network_info(text: str) -> str:
    try:
        t = str(text)
        t = re.sub(r"(host=')([^']+)(')", r"\1<hidden-host>\3", t)
        t = re.sub(r'(host=")([^"]+)(")', r'\1<hidden-host>\3', t)
        t = re.sub(r"(port=)(\d+)", r"\1<hidden-port>", t)
        t = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<hidden-ip>", t)
        t = re.sub(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b", "<hidden-ipv6>", t)
        t = re.sub(r"\b::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}\b", "<hidden-ipv6>", t)
        t = re.sub(r"(Connection to )([^\s]+)", r"\1<hidden-host>", t)
        hide_url_hosts = os.environ.get("RUNNODE_HIDE_URL_HOSTS", "true").lower() != "false"
        if hide_url_hosts:
            t = re.sub(r"(https?://)([^/\s]+)", r"\1<hidden-host>", t)
        return t
    except Exception:
        return str(text)

def generate_request_id(task_type: str, provider: str) -> str:
    short_uuid = str(uuid.uuid4())[:8]
    return f"rn_{provider}_{task_type}_{short_uuid}"

def log_prepare(task_name, request_id, prefix, service_name, model_version=None, speed=None, **kwargs):
    info = f" {kwargs}" if kwargs else ""
    if model_version: info += f" model_version={model_version}"
    if speed: info += f" speed={speed}"
    print(f"{PREFIX} {prefix} [{task_name}] {request_id} Preparing...{info}")

def log_complete(task_name, request_id, prefix, service_name, image_url=None, **kwargs):
    info = f" {kwargs}" if kwargs else ""
    if image_url: info += f" image_url={image_url}"
    print(f"{PREFIX} {prefix} [{task_name}] {request_id} Completed.{info}")

def log_error(message, request_id=None, detail=None, source="RunNode", service_name=None):
    if service_name:
        print(f"{ERROR_PREFIX} [{source}] {service_name} Error: {message} - {str(detail)}")
    else:
        print(f"{ERROR_PREFIX} [{source}] Error: {message} - {str(detail)}")

class ProgressBar:
    def __init__(self, request_id, service_name, extra_info="", streaming=True, task_type="Task", source="RunNode"):
        self.request_id = request_id
        self.service_name = service_name
        self.streaming = streaming
        self.last_update = time.time()
    
    def update_absolute(self, value):
        if time.time() - self.last_update > 0.5:
            print(f"{PROCESS_PREFIX} Progress: {value}%")
            self.last_update = time.time()
            
    def update(self, value):
        self.update_absolute(value)

    def set_generating(self):
        if self.streaming:
            print(f"{PROCESS_PREFIX} Generating...")

    def error(self, message):
        print(f"{ERROR_PREFIX} Error: {sanitize_sensitive_network_info(str(message))}")
        
    def done(self, char_count=0, elapsed_ms=0):
        print(f"{PREFIX} Done in {elapsed_ms}ms")

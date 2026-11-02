"""检查当前虚拟环境，并以固定本机地址运行页面。"""

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
ADDRESS = "127.0.0.1"
PORT = 8501
_MODULES = {
    "langchain-openai": "langchain_openai",
    "langchain-core": "langchain_core",
    "langgraph-checkpoint-sqlite": "langgraph.checkpoint.sqlite",
}


def check_environment():
    """只检查依赖和端口，不读取密钥或初始化账户。"""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("请使用项目虚拟环境中的 Python 运行，例如 .\\.venv\\Scripts\\python.exe run_local.py。")
    for line in (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        package, expected = line.strip().split("==")
        try:
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"依赖 {package} 需要 {expected}，当前为 {actual}；请安装 requirements.txt。")
            importlib.import_module(_MODULES.get(package, package.replace("-", "_")))
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError(f"依赖 {package} 不可用，请在虚拟环境中安装 requirements.txt。") from error
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((ADDRESS, PORT))
        except OSError as error:
            raise RuntimeError("本机 8501 端口已被占用或不可用，请先停止已有服务；不会改用其他端口。") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="数析本机启动器，固定使用 http://127.0.0.1:8501")
    parser.add_argument("--check", action="store_true", help="只检查虚拟环境、依赖和端口")
    arguments = parser.parse_args(argv)
    try:
        check_environment()
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"启动检查失败：{error}\n")
    if arguments.check:
        print("启动检查通过：虚拟环境和依赖可用，本机 8501 端口空闲。")
        return 0
    command = [sys.executable, "-m", "streamlit", "run", str(PROJECT_DIR / "app.py"),
               "--server.address", ADDRESS, "--server.port", str(PORT), "--server.headless", "true"]
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    print(f"正在启动数析：http://{ADDRESS}:{PORT}；在此终端按 Ctrl+C 停止。", flush=True)
    try:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, **options)
    except OSError as error:
        parser.exit(1, f"无法启动页面：{error}\n")
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        return 130
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())


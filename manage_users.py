"""可信本机终端的账号初始化与密码重置入口。"""

import argparse
import getpass
import os

import auth_store


def _read_password():
    password = getpass.getpass("请输入密码（12 至 128 个字符，输入不会显示）：")
    repeated = getpass.getpass("请再次输入密码：")
    if password != repeated:
        raise ValueError("两次密码不一致，请重新运行。")
    return password


def _prepare_storage_key():
    import secure_storage

    if "SHUXI_STORAGE_KEY" in os.environ:
        secure_storage.load_cipher()
        return
    key_path = secure_storage.DEFAULT_KEY_PATH
    user_directory = secure_storage.DEFAULT_DATA_DIR / "users"
    if not key_path.exists() and user_directory.exists() and any(user_directory.iterdir()):
        raise secure_storage.StorageError("已有用户资料但存储密钥缺失，请恢复原密钥；不能生成替代密钥。")
    secure_storage.initialize_key()
    secure_storage.load_cipher()


def main(argv=None):
    parser = argparse.ArgumentParser(description="数析本机账号管理；密码通过隐藏输入提供。")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="首次初始化存储密钥和管理员")
    initialize.add_argument("--username", required=True, help="管理员用户名（小写字母开头，3 至 32 位）")
    reset = commands.add_parser("reset-password", help="重置账号密码，并使其旧会话失效")
    reset.add_argument("username", help="已有账号的用户名")
    arguments = parser.parse_args(argv)

    import secure_storage

    try:
        if arguments.command == "init" and auth_store.has_users():
            raise ValueError("账号库已经初始化，请在页面中登录并管理账号。")
        password = _read_password()
        auth_store._validate_credentials(arguments.username, password)
        if arguments.command == "init":
            _prepare_storage_key()
            auth_store.bootstrap_admin(arguments.username, password)
            print("管理员与本机存储密钥初始化完成，可以启动页面登录。")
        else:
            auth_store.reset_password(arguments.username, password)
            print("密码已重置，该账号的旧会话已全部撤销。")
    except (ValueError, PermissionError, secure_storage.StorageError) as error:
        parser.exit(1, f"操作失败：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


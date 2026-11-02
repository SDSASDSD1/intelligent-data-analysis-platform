"""数析停机备份和恢复命令行入口。"""

import argparse

from backup_store import BackupError, create_backup, restore_backup


def main(argv=None):
    parser = argparse.ArgumentParser(description="请先停止数析，再备份或恢复；原存储密钥须单独保管。")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="备份账户与用户加密资料")
    create.add_argument("archive", help="新备份 ZIP 路径")
    create.add_argument("--key-path", help="原存储密钥文件路径，不能传入密钥内容")
    restore = commands.add_parser("restore", help="恢复到不存在或为空的资料目录")
    restore.add_argument("archive", help="已有备份 ZIP 路径")
    restore.add_argument("--destination", help="恢复目标；默认为项目的 private_data")
    restore.add_argument("--key-path", help="原存储密钥文件路径，不能传入密钥内容")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "create":
            output = create_backup(arguments.archive, key_path=arguments.key_path)
            print(f"备份完成：{output}\n备份不含存储密钥，请单独保管原密钥；恢复后须重新登录。")
        else:
            output = restore_backup(arguments.archive, destination=arguments.destination, key_path=arguments.key_path)
            print(f"恢复完成：{output}\n请使用原存储密钥启动并重新登录；恢复审核时需重新上传原文件。")
    except (BackupError, RuntimeError, OSError) as error:
        parser.exit(1, f"操作失败：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


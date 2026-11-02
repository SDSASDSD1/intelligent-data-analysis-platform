"""本机登录与账户管理页面；身份始终由账户数据库确认。"""

import sqlite3

import streamlit as st

import auth_store
import secure_storage


def _clear_session(token=None, user_id=None):
    st.session_state.clear()
    if token is not None:
        st.session_state["auth_token"] = token
        st.session_state["active_user_id"] = user_id


def require_user():
    """未登录返回 None，让主页面在展示私人内容前停止。"""
    try:
        initialized = auth_store.has_users()
        token = st.session_state.get("auth_token")
        user = auth_store.resolve_session(token) if token else None
    except (sqlite3.Error, OSError):
        st.error("账户库无法读取，请检查本机资料目录。")
        return None
    if not initialized:
        _clear_session()
        st.info("首次使用：在项目终端运行 .\\.venv\\Scripts\\python.exe manage_users.py init --username admin，设置管理员后刷新页面。")
        return None
    if user is None:
        if token:
            _clear_session()
            st.warning("登录已失效，请重新登录。")
        with st.container(key="panel_login"):
            st.subheader("登录智能数据分析平台")
            st.caption("登录你的工作台，继续查看数据与分析记录。")
            with st.form("login_form", clear_on_submit=True, border=False):
                username = st.text_input("用户名", key="login_username", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", key="login_password", placeholder="请输入密码")
                submitted = st.form_submit_button("登录", key="login_submit", type="primary", width="stretch")
            st.caption("账号由管理员创建，分析资料按账号独立保存。")
        if submitted:
            token = auth_store.authenticate(username, password)
            if token is None:
                st.error("登录失败：请检查用户名、密码或账户状态；连续失败后需稍等再试。")
            else:
                user = auth_store.resolve_session(token)
                _clear_session(token, user["id"])
                st.rerun()
        return None
    if st.session_state.get("active_user_id") != user["id"]:
        _clear_session(token, user["id"])
        st.rerun()
    with st.sidebar:
        with st.container(key="panel_account"):
            st.caption("我的工作台")
            role_label = {"admin": "管理员", "analyst": "分析员"}[user["role"]]
            st.write(f"当前账户：{user['username']}（{role_label}）")
            if st.button("退出登录", key="logout", width="stretch"):
                auth_store.logout(token)
                _clear_session()
                st.rerun()
    return user


def workspace(user):
    return secure_storage.workspace_for_user(user["id"])


def render_account_management(user):
    if user["role"] != "admin":
        return
    with st.sidebar.expander("账户管理"):
        token = st.session_state["auth_token"]
        with st.form("create_account", clear_on_submit=True):
            username = st.text_input("新用户名（小写字母开头，3～32 位）", key="new_username")
            password = st.text_input("新密码（12～128 位）", type="password", key="new_password")
            role = st.selectbox("角色", ["analyst", "admin"],
                                format_func=lambda value: {"analyst": "分析员", "admin": "管理员"}[value],
                                key="new_role")
            create = st.form_submit_button("创建账户", key="create_user")
        if create:
            try:
                auth_store.create_user(token, username, password, role)
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("账户已创建。")
        try:
            users = auth_store.list_users(token)
        except PermissionError:
            st.warning("管理权限已失效，请重新登录。")
            return
        labels = {item["id"]: f"{item['username']} · {item['role']} · {'启用' if item['enabled'] else '停用'}"
                  for item in users}
        with st.form("account_status"):
            target_id = st.selectbox("选择账户", list(labels), format_func=labels.get, key="manage_user_id")
            enabled = st.selectbox("设置状态", [True, False],
                                   format_func=lambda value: "启用" if value else "停用", key="manage_enabled")
            update = st.form_submit_button("更新账户状态", key="update_user")
        if update:
            try:
                auth_store.set_user_enabled(token, target_id, enabled)
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.rerun()


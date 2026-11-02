"""下载按钮：浏览器点击后才向经过认证的本机端点索取文件。"""

import sqlite3

import streamlit as st

from download_service import DownloadStore, start_server


DOWNLOAD_JS = """
export default function({ data, parentElement }) {
  const button = parentElement.querySelector("button");
  const status = parentElement.querySelector("[role=status]");
  button.textContent = data.label;
  const controller = new AbortController();
  button.onclick = async () => {
    button.disabled = true;
    status.textContent = "正在下载…";
    try {
      const response = await fetch(data.endpoint, {
        method: "POST",
        credentials: "omit",
        cache: "no-store",
        headers: { "Authorization": "Bearer " + data.token },
        signal: controller.signal
      });
      if (!response.ok) {
        throw new Error(response.status === 404
          ? "下载已过期，请重新运行页面后再试。"
          : "下载失败，请检查登录状态并刷新页面。");
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = data.filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status.textContent = "文件已交给浏览器保存。";
    } catch (error) {
      if (error.name !== "AbortError") {
        status.textContent = "下载失败，请刷新页面或重新登录。";
      }
    } finally {
      button.disabled = false;
    }
  };
  return () => { controller.abort(); button.onclick = null; };
}
"""


def _close_service(resource):
    resource[1].shutdown()
    resource[1].server_close()


@st.cache_resource(on_release=_close_service)
def get_service():
    store = DownloadStore()
    return store, start_server(store)


def render_download(label, content, *, file_name, mime, key, kind="report", approved=False):
    token = st.session_state.get("auth_token")
    if not token:
        st.warning("请登录后下载。")
        return
    try:
        store, server = get_service()
        artifact_id = store.register(token, content, file_name, mime, kind, approved)
    except (PermissionError, ValueError, OSError, sqlite3.Error):
        st.warning("下载暂不可用，请检查登录状态；文件过大或缓存已满时请稍后重试。")
        return
    component = st.components.v2.component(
        "shuxi_secure_download",
        html='<button type="button"></button><span role="status" aria-live="polite"></span>',
        css="""button { padding: .5rem .8rem; border-radius: .5rem; cursor: pointer;
          color: var(--st-text-color); background: var(--st-background-color);
          border: 1px solid var(--st-border-color); font: inherit; }
          button:disabled { opacity: .6; cursor: wait; }
          span { margin-left: .6rem; font-size: .85rem; }""",
        js=DOWNLOAD_JS,
    )
    component(
        data={"endpoint": f"http://127.0.0.1:{server.server_address[1]}/download/{artifact_id}",
              "token": token, "filename": file_name, "label": label},
        key=key,
    )


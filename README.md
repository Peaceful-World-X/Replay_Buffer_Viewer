# Replay Buffer Viewer

用于在浏览器中查看 RLinf replay buffer 轨迹，包括多相机画面、状态曲线、reward、干预标记和轨迹数据结构。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone <项目仓库地址>
cd replay_buffer_viewer

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install flask numpy av torch
```

Windows PowerShell 中激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 启动

在项目目录中运行：

```bash
source .venv/bin/activate
python app.py
```

浏览器打开：

```text
http://127.0.0.1:8088
```

可通过环境变量修改监听地址和端口：

```bash
REPLAY_VIEWER_HOST=127.0.0.1 REPLAY_VIEWER_PORT=9000 python app.py
```

## 使用

在页面顶部输入以下任一种路径，然后点击“打开数据”：

- 单个 RLinf 轨迹 `.pt` 文件的绝对路径。
- replay buffer 目录的绝对路径；目录中需要包含 `trajectory_index.json` 和对应的轨迹文件。

载入后可以播放多相机画面、逐帧查看数据、切换状态曲线并查看 reward。页面默认剔除 `chunk_valid_mask` 标记的无效帧，可通过工具栏复选框关闭。

服务端必须能够直接访问所填写的数据路径。首次生成视频缓存时可能需要等待片刻，缓存文件位于项目的 `cache/` 目录。

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
python -m pip install flask numpy av torch pyarrow
```

Windows PowerShell 中激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 启动

在项目目录中运行：

```bash
source /16T-2/sudu/code/RLinf/.venv/bin/activate
python app.py
```

当前机器也可直接运行 `./run.sh`，脚本优先使用项目 `.venv`，其次使用已有的 `/16T-2/sudu/code/RLinf/.venv`；可用 `REPLAY_VIEWER_PYTHON` 指定 Python。

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

- 原始 LeRobot v3 数据集目录（包含 `meta/info.json`、episode 元数据、parquet 和视频）。
- 单个 RLinf 轨迹 `.pt` 文件的绝对路径。
- replay buffer 目录的绝对路径；目录中需要包含 `trajectory_index.json` 和对应的轨迹文件。

载入后可以播放多相机画面、逐帧查看数据、切换状态曲线并查看 reward。页面默认剔除 `chunk_valid_mask` 标记的无效帧，可通过工具栏复选框关闭。

左侧用“含人工介入”标记对应轨迹；选择“只看含人工介入”可筛选并跳转到匹配轨迹，列表同时显示匹配数量。

其下的结果筛选支持“全部结果／只看成功／只看失败”，可与人工介入筛选和搜索叠加使用，两个标注模式都适用。未知结果只出现在“全部结果”中。切换筛选时，如果当前轨迹不符合条件，会跳转到第一条匹配轨迹；打开新数据集时重置两个筛选。

左侧每条轨迹及顶部文件信息显示结果标签：**绿色“成功”、红色“失败”、灰色“未知”**；列表上方统计当前筛选结果中的三类数量。结果标签与“含人工介入”可同时显示，悬停可查看判断依据。原始 LeRobot 使用 `episode_success` 标注；RLinf replay 使用末个有效帧的终止奖励（1=成功、0=失败）。没有终止结果、被截断、奖励未标注或不是 0/1 时显示未知，不把中途裁剪片段判为失败，也不以较早出现的成功奖励代表片段末端结果。

播放时间轴、人工裁剪时间轴和曲线背景都显示控制阶段：**紫色为模型推理，蓝色为人工介入，灰色为未知**。色块标出起止帧，悬停显示时间范围，点击可跳转；视频上方同步显示当前控制方式。绿色保留区间和橙色待保存选区位于独立行，不覆盖控制阶段。

阶段判断优先使用逐帧 `is_intervention` / `intervene_flags`，不依赖 reward 是否存在；原始数据缺少逐帧标记时，再使用声明的采集策略 codebook 或人工接管区间。缺少控制标记时显示未知，不自动归为推理。`zd_10` 的 `complementary_info.phase` 表示扎带插入关键操作区间，不表示模型/人工控制切换；Episode 0、3 没有人工介入，会显示完整的模型推理色带。

支持内嵌 RGB／H.264 轨迹，以及使用 `curr_obs.video_frame_indices` 引用外部视频的 G1 raw replay 缓存。后者按同级或上一级 `source_manifest.json` 的 `camera_order`、`media` 相对路径和 FPS 关联前置及双腕视频，不需要把图像写回 `.pt`。没有 `media` 列表的合并缓存则读取 `packaged_source_dataset` 指向的 LeRobot v3 元数据，用 `forward_inputs.episode_index` 定位各相机的 chunk/file，并将 `frame_index` 与视频偏移交叉核对；使用合并后的 episode 编号，支持不同相机分文件及指向原视频的软链接。例如可直接打开：

```text
/16T-2/sudu/code/.data/RLinf/zd_10_rlt_cache/raw_replay
/16T-2/sudu/code/.result2/RLT/RLinf/zd89_20261027/source_cache/raw_replay
```

复制缓存时请一起保留 `source_manifest.json`、`raw_replay/` 和 `dataset/` 的相对目录结构。视频仅包含当前轨迹引用的帧，与状态、动作、reward 和接管标记对齐。没有 `chunk_valid_mask` 时所有帧都有效，过滤开关自动禁用。缺少视频或 manifest 时仍可播放数值时间轴、查看曲线和逐帧数据，页面会提示原因；无法确定 FPS 时数值回放暂按 30 帧/秒。

服务端必须能够直接访问所填写的数据路径。首次生成视频缓存时可能需要等待片刻，缓存文件位于项目的 `cache/` 目录。

## 人工检查与裁剪原始数据

打开 `/16T-2/sudu/code/.data/RLinf/zd_10` 后，左侧列出全部 5 个完整 episode（共 4948 帧）。

1. 选择 episode，播放三路视频或逐帧检查。点击“人工接管段”的帧号可直接跳转到原始标注的接管开始位置。
2. 在可用片段第一帧点击“标记开始”，默认保留至 episode 末尾；需要提前结束时再点击“标记结束”。也可直接输入从 0 开始的帧号。至少选择 2 帧，**首尾两帧都会保留**。
3. 选择“可用介入／可用动作／纠错恢复”，填写可选备注，用“预览选区”检查，然后点击“保存保留片段”。同一 episode 可保存多个不重叠片段。实际接管标记按原始数据保留。末端结果默认“沿用原始标注”，只有人工确认后才选择成功或失败。
4. 点击“导出并覆盖训练轨迹”，将当前数据集内**全部已保存片段**导出到固定子目录 `zd_10/training_trajectories/`。界面显示进度、帧数、transition 数和结果路径。
5. 点击顶部“预览训练轨迹”，或直接打开该子目录，查看已导出的 RLinf 轨迹及三路裁剪视频。
6. 点击“返回原始数据重新裁剪”，回到对应的完整 episode 并载入该片段起止点。也可在原始数据的已保存列表中点击“修改起止点”。调整后点击“保存修改”，再导出覆盖同一子目录。删除片段后再次导出会同步清理旧轨迹；删除全部片段后可以导出空集合。

没有可用片段的 episode 可点击“标记检查完成”。导出使用暂存目录，全部生成并检查成功后才替换旧导出；生成失败、标注在导出期间改变或替换失败时保留旧结果。不会覆盖同名但未经此查看器创建的目录。训练目录是当前已保存标注的完整快照，不是追加历史版本；训练任务读取期间请避免重新导出。

键盘快捷键：`I` 标记开始，`O` 标记结束，空格播放／暂停，左右方向键逐帧移动；输入框编辑时不触发快捷键。未保存的选区在当前浏览器中保留，已保存标注写入原数据集根目录的 `review_annotations.json`，刷新或重启服务后仍存在。并发修改会检查标注版本，源文件变化后旧标注会锁定以避免错裁。

## 导出内容与 RLinf 读取

`training_trajectories/` 同时包含以下内容：

- `trajectory_*.pt`、`trajectory_index.json`、`metadata.json`：RLinf replay checkpoint。每个保留区间生成一个独立轨迹，包含动作、奖励、结束标记、原始接管标记、当前/下一状态和源帧索引。
- `meta/`、`data/`、`videos/`：裁剪后的 LeRobot v3 数据集，保留选中的全部帧，可供后续图像或特征预处理使用。
- `source_manifest.json`：每条轨迹对应的相机、视频和帧索引；`review_selection.json`：原始 episode、范围、人工标签及备注；`training_export.json`：导出版本和返回原始目录的信息。

选择 `[开始帧,结束帧]` 共 N 帧，会生成 **N−1 条 transition**：第 t 帧的真实动作及当前状态，对应第 t+1 帧的真实下一状态。最后一帧只作为 `next_obs`，不伪造下一状态，不跨片段拼接。视频保留全部 N 帧，训练预览显示 N−1 个当前状态。JSON 中范围统一使用 `[start_frame,end_frame)`。

奖励优先使用源数据已记录的字段，否则按源 `rl_intervals` 的成功/失败阶段末端生成稀疏奖励和终止标记；未知区域填 0，并通过 `forward_inputs.reward_is_annotated=False` 区分未标注。手动裁剪本身不代表成功或环境终止。“人工确认成功／失败”仅覆盖最后一条 transition 的奖励为 1／0，并标记终止。

这些是**原始观测 replay**，`curr_obs` / `next_obs` 保存 `states`、`source_index`、`video_frame_indices`；没有模型推理，因而不会生成 `z_rl`、`proprio`、`ref_chunk`、策略 logprob 或 value。它符合本机 RLinf replay 存储读取接口，但需要这些特征的 RLT 训练仍须先运行对应特征预处理。视频索引由查看器解析，通用 replay sampler 本身不会解码图像。

在 RLinf 环境中读取示例：

```python
from rlinf.data.storage.replay import TrajectoryReplayBuffer

buffer = TrajectoryReplayBuffer(
    enable_cache=True, cache_size=5, sample_window_size=5, auto_save=False,
)
buffer.load_checkpoint("/16T-2/sudu/code/.data/RLinf/zd_10/training_trajectories")
batch = buffer.sample(32)
buffer.close()
```

LeRobot 导出中每个区间成为独立 episode，重新编号 `episode_index`、`frame_index`、`index` 和时间戳，并重建数据统计及视频偏移。动作、状态、接管标记逐帧复制；视频按精确帧边界裁剪并重新编码为 H.264（CRF 18）。原始人工接管和阶段区间会裁剪并平移；未完整保留 episode 的整体成功标签设为 `unknown`，截断成功阶段终点时其 outcome 也设为 `unknown`。图像统计从每段最多 32 帧缩放到 64×64 后计算。人工末端结果保存在裁剪清单和 replay 中，不重写源阶段标签。原始 parquet 和视频文件不被裁剪或覆盖。

修改代码后需重启 `python app.py` 并刷新页面。回归检查：`python -m unittest -v test_app test_review`。

## Actor 门控：按开启／关闭两帧标注完整轨迹

这项标注用于训练“是否开启 actor”的分类器。打开**完整原始 LeRobot v3 数据集**，例如：

```text
/16T-2/sudu/code/.data/RLinf/zd_89_261027
/16T-2/sudu/code/.result2/RLT/RLinf/zd89_20261027/source_cache/dataset
```

视频上方的模式栏显示当前模式，点击同一个切换按钮即可在“Actor 起止标定”和“片段裁剪”之间切换，每次只显示对应标注面板。默认进入 Actor 起止标定，并记住上次选择。切换时视频位置、未保存草稿和已保存标注保留；只有当前模式的标注快捷键生效，空格和方向键仍可控制视频。

选择“Actor 起止标定”后，在视频下方使用“Actor 门控 · 开启与关闭标注”：

1. 播放或逐帧定位到扎带接近孔口、应该开始使用 actor 的第一帧，点击“当前帧设为开启”，也可按 `[`。
2. 定位到已经确认成功穿过、应该关闭 actor 的第一帧，点击“当前帧设为关闭”，也可按 `]`。
3. 界面分别显示开启、关闭两帧的三路相机图片；点击缩略图可打开原尺寸图片核对，也可直接输入帧号、点击时间轴跳转。
4. 点击“保存标注并自动导出”。每条 episode 保存一对边界，可再次调整并保存。保存的标签在刷新和重启后仍保留。

标签区间为 **`[开启帧, 关闭帧)`**：开启帧标1，关闭帧本身以及其后标0。例如开启帧100、关闭帧350，则帧100～349标1，其他帧标0。**不限制100帧或200帧，也不自动把末尾一段视为正样本。** 接近至穿过期间的重试是否保持开启，由人工选择的边界决定。

“确认整条均为0”仅用于人工确认没有任何 actor 开启阶段的完整轨迹。“移除此条门控标注”会恢复为未标注，并从导出中排除；未标注轨迹不会自动作为负样本。旧的动作裁剪标注不会自动转换为门控标签。

标注保存为原数据集目录下的 `actor_gate_annotations.json`。每次保存、修改或删除后，后台自动更新同目录的 `actor_gate_training/`，它是**全部已人工标注 episode 的完整快照**：

| 文件 | 内容 |
|---|---|
| `labels_manifest.json` | 原始数据集路径、episode编号、开/关边界、帧数、来源指纹、相机顺序和原视频路径 |
| `labels/episode_*.npz` | 完整 episode 的 `labels`、`state`、`frame_index`、`source_index`、`timestamp` 和 `video_frame_indices` |
| `data/episode_*.parquet` | 完整 episode 的原始数值列，新增 `annotation.actor_gate`，不改变动作、reward、人工介入等字段 |
| `images/episode_*_{start,end}_*.jpg` | 人工选择的开/关两帧的各路相机原尺寸图片 |
| `frame_labels.csv` | 所有已标注 episode 的逐帧0/1标签，可从界面下载 |
| `gate_export.json`、`actor_gate_annotations.json` | 导出版本及对应标注快照 |

**原始视频不复制、不裁剪、不覆盖**。训练时用清单中的原视频路径和 NPZ 中的逐帧视频索引取图；移动数据时需同步更新源路径。该目录是门控分类数据包，不是动作 replay，也不是独立可迁移的 LeRobot 视频数据集。现有按成功片段生成标签的训练准备入口应改读此清单，不能再用旧成功裁剪区间覆盖人工门控标签。

读取数值与标签示例：

```python
import json
from pathlib import Path
import numpy as np

root = Path('/path/to/original_dataset/actor_gate_training')
manifest = json.loads((root / 'labels_manifest.json').read_text())
for episode in manifest['episodes']:
    with np.load(root / episode['labels_file']) as data:
        labels = data['labels']             # [N]，关闭帧为0
        state = data['state']               # [N, state_dim]，完整原始状态
        source_index = data['source_index'] # 原始数据行索引
        video_frames = data['video_frame_indices']  # [N, cameras]
    # 按 episode 划分训练/验证，不能随机拆分相邻帧。
    # 按 episode['camera_order'] 和 episode['media'] 读取对应图像。
```

导出在暂存目录中完成，成功后替换本查看器拥有的旧导出；失败时旧导出保留，标注仍已保存，可点击“重新导出已标注轨迹”。源数据变化或并发旧版本写入会被拒绝。训练进程读取期间不要重新导出同一目录。

回归检查：`python -m unittest -v test_gate test_app test_review`。

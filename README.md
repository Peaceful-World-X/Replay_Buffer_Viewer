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
source .venv/bin/activate
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

/* Gate intervals are [ON observation, OFF observation); originals stay complete. */
const Gate = (() => {
  const el = id => document.getElementById(id);
  let state = null, key = '', generation = 0, busy = false, pollTimer = null, imageTimer = null;
  let dirty = false;
  function message(text, error = false) {
    el('gateMessage').textContent = text;
    el('gateMessage').classList.toggle('error', error);
  }
  function storageKey() { return `actor-gate-draft:${meta.path}:${meta.episodeId}:${key}`; }
  function interval() {
    if (el('gateStart').value === '' || el('gateEnd').value === '') throw Error('请分别标记开启和关闭两帧。');
    const start = Number(el('gateStart').value), end = Number(el('gateEnd').value);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end >= meta.totalFrames)
      throw Error('要求 0 ≤ 开启帧 < 关闭帧 < 总帧数；关闭帧标为0。');
    return { start, end };
  }
  function forgetDraft() { try { localStorage.removeItem(storageKey()); } catch (_) {} }
  function restore() {
    el('gateStart').value = state.entry?.start_frame ?? '';
    el('gateEnd').value = state.entry?.end_frame ?? '';
    el('gateNote').value = state.entry?.note ?? '';
    dirty = false;
  }
  function persist() {
    dirty = true;
    try { localStorage.setItem(storageKey(), JSON.stringify({start: el('gateStart').value,
      end: el('gateEnd').value, note: el('gateNote').value, savedAt: state.entry?.updated_at ?? ''})); } catch (_) {}
    render();
  }
  function images() {
    if (!state) return;
    for (const [field, id] of [['gateStart', 'gateStartImages'], ['gateEnd', 'gateEndImages']]) {
      const box = el(id), value = el(field).value, frame = Number(value), stamp = `${key}:${value}`;
      if (box.dataset.frame === stamp) continue;
      box.dataset.frame = stamp; box.replaceChildren();
      if (value === '' || !Number.isInteger(frame) || frame < 0 || frame >= meta.totalFrames) {
        box.textContent = '标记后显示这一帧的三路图片'; continue;
      }
      for (const camera of ['left', 'main', 'right'].filter(c => meta.cameras.includes(c))) {
        const item = document.createElement('figure'), img = document.createElement('img'), caption = document.createElement('figcaption');
        caption.textContent = `${{left:'左腕',main:'前置',right:'右腕'}[camera]} · 帧 ${frame}`;
        img.alt = caption.textContent; img.src = `/api/gate/image/${camera}?key=${key}&frame=${frame}`;
        img.onerror = () => { caption.textContent += ' · 图片读取失败'; };
        const link = document.createElement('a'); link.href = img.src; link.target = '_blank'; link.rel = 'noopener'; link.title = '打开原尺寸图片'; link.append(img); item.append(link, caption); box.append(item);
      }
      if (!box.children.length) box.textContent = '此 episode 缺少相机视频';
    }
  }
  function render() {
    if (!state) return;
    el('gateControls').disabled = busy || !!state.entry?.stale;
    el('gateRemove').disabled = !state.entry;
    const track = el('gateTrack'); track.replaceChildren();
    let selected = null;
    try { selected = interval(); } catch (e) { el('gateSummary').textContent = state.entry?.mode === 'all_off' && !dirty ? '已人工确认：完整轨迹全为0。' : e.message; }
    el('gateSave').disabled = !selected;
    el('gateJumpStart').disabled = el('gateStart').value === '';
    el('gateJumpEnd').disabled = el('gateEnd').value === '';
    if (selected) {
      el('gateSummary').textContent = `${dirty?'待保存':'已保存'}：帧 ${selected.start}～${selected.end-1} 为1，共 ${selected.end-selected.start} 帧；帧 ${selected.end} 起为0。完整轨迹 ${meta.totalFrames} 帧都会导出。`;
      const range = document.createElement('div'); range.className = dirty ? 'gate-on draft' : 'gate-on';
      range.style.left = `${selected.start/meta.totalFrames*100}%`;
      range.style.width = `${(selected.end-selected.start)/meta.totalFrames*100}%`; range.textContent = 'Actor 开启 = 1';
      track.append(range);
    }
    track.classList.toggle('gate-unknown', !selected && (dirty || !state.entry));
    const cursor = document.createElement('span'); cursor.id = 'gateCursor'; cursor.className = 'review-cursor'; track.append(cursor);
    el('gateDownload').classList.toggle('hidden', !state.exportCurrent || busy);
    el('gateDownload').href = `/api/gate/download/csv?key=${key}`;
    const reviewed = new Set(state.reviewedEpisodes);
    trajectoryCollection.filter(item => item.path === meta.path).forEach(item => item.gateReviewed = reviewed.has(item.episodeId));
    renderTrajectoryList(); position(timelineTime);
    clearTimeout(imageTimer); imageTimer = setTimeout(images, 150);
  }
  function beginLoad() {
    generation++; state = null; key = ''; busy = false; dirty = false;
    clearTimeout(pollTimer); clearTimeout(imageTimer);
    el('gatePanel').classList.add('hidden'); el('gateControls').disabled = true;
    ['gateStartImages','gateEndImages'].forEach(id => {el(id).replaceChildren(); delete el(id).dataset.frame;});
  }
  async function load() {
    if (meta.sourceType !== 'lerobot') return;
    const current = generation; key = meta.key;
    el('gatePanel').classList.remove('hidden'); message('正在读取门控标注…'); el('gateExportMessage').textContent = '';
    try {
      const result = await api(`/api/gate?key=${key}`);
      if (generation !== current) return;
      state = result; restore();
      try {
        const draft = JSON.parse(localStorage.getItem(storageKey()) || 'null');
        if (draft && draft.savedAt === (state.entry?.updated_at ?? '')) {
          el('gateStart').value = draft.start; el('gateEnd').value = draft.end; el('gateNote').value = draft.note; dirty = true;
        }
      } catch (_) {}
      ['gateStart','gateEnd'].forEach(id => el(id).max = meta.totalFrames-1);
      message(state.entry?.stale ? '源数据已变化，旧标注已锁定，请先核对。' : `已标注 ${state.reviewedEpisodes.length} 条轨迹。标注文件：${state.annotationPath}`, !!state.entry?.stale);
      if (state.jobId) { busy = true; poll(state.jobId, current); }
      else el('gateExportMessage').textContent = state.exportCurrent ? `导出已同步：${state.exportDirectory}` : '尚未导出最新标注。保存两帧后会自动导出。';
      render();
    } catch (e) { if (generation === current) message(e.message, true); }
  }
  function capture(field) {
    if (!state || busy || state.entry?.stale) return;
    const frame = frameAtTime(timelineTime); pausePlayback(); seekTo(frame/meta.fps);
    el(field).value = frame; persist();
    message(`已选择${field === 'gateStart' ? '开启' : '关闭'}图片：Episode ${meta.episodeId}，帧 ${frame}。`);
  }
  async function save(mode = 'interval', remove = false) {
    if (!state || busy) return;
    const current = generation, body = {revision: state.revision, mode, note: el('gateNote').value};
    try {
      if (mode === 'interval' && !remove) {const s = interval(); body.startFrame = s.start; body.endFrame = s.end;}
      busy = true; render();
      const result = await api(`/api/gate?key=${key}`, {method: remove?'DELETE':'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if (generation !== current) return;
      forgetDraft(); state = result; restore();
      message(remove ? '已移除标注：该轨迹恢复为未标注，不再参与导出。' : '标注已保存，正在自动导出完整轨迹…');
      render(); poll(result.jobId, current);
    } catch (e) {if (generation === current) {busy = false; message(e.message, true); render();}}
  }
  async function poll(id, current) {
    if (generation !== current) return;
    try {
      const job = await api(`/api/review/export/${id}`);
      if (generation !== current) return;
      el('gateExportMessage').textContent = job.message;
      if (job.status === 'running') { pollTimer = setTimeout(() => poll(id, current), 500); return; }
      busy = false;
      if (job.status === 'complete') {
        const refreshed = await api(`/api/gate?key=${key}`);
        if (generation !== current) return;
        state = refreshed;
        const r = job.result;
        el('gateExportMessage').textContent = `导出完成：${r.episodes} 条完整轨迹，${r.frames} 帧（1：${r.positive_frames} 帧，0：${r.negative_frames} 帧）。目录：${r.directory}`;
      } else message(`标注已保存，但导出失败：${job.message}。可点击“重新导出”。`, true);
      render();
    } catch(e) {if (generation === current) {busy = false; message(e.message, true); render();}}
  }
  function position(time) {
    if (!state) return;
    const frame = frameAtTime(time); let label = '未标注';
    try {const s = interval(); label = frame >= s.start && frame < s.end ? 'Actor 开启 = 1' : 'Actor 关闭 = 0';}
    catch (_) {if (state.entry?.mode === 'all_off' && !dirty) label = 'Actor 关闭 = 0';}
    el('gatePosition').textContent = `帧 ${frame} · ${label}${dirty?'（待保存）':''}`;
    const cursor = el('gateCursor'); if (cursor) cursor.style.left = `${frame/Math.max(1,meta.totalFrames-1)*100}%`;
  }
  document.addEventListener('DOMContentLoaded', () => {
    el('gateMarkStart').onclick = () => capture('gateStart'); el('gateMarkEnd').onclick = () => capture('gateEnd');
    ['gateStart','gateEnd','gateNote'].forEach(id => el(id).addEventListener('input',persist));
    el('gateSave').onclick = () => save();
    el('gateAllOff').onclick = () => {if (confirm('确认这条完整轨迹没有任何需要开启 actor 的阶段？保存后全帧标为0。')) save('all_off');};
    el('gateRemove').onclick = () => {if (confirm('移除此条门控标注，并从训练导出中排除该轨迹？')) save('interval',true);};
    el('gateResetDraft').onclick = () => {forgetDraft(); restore(); render();};
    for (const [button,field] of [['gateJumpStart','gateStart'],['gateJumpEnd','gateEnd']]) el(button).onclick = () => {pausePlayback();seekTo(Number(el(field).value)/meta.fps);};
    el('gateTrack').onclick = event => {const rect=el('gateTrack').getBoundingClientRect();pausePlayback();seekTo(Math.floor((event.clientX-rect.left)/rect.width*meta.totalFrames)/meta.fps);};
    el('gateExport').onclick = async () => {
      if (busy || !state) return;
      const current = generation;busy = true;render();
      try {const result=await api(`/api/gate/export?key=${key}`,{method:'POST'});if(generation===current)poll(result.jobId,current);}
      catch(e){if(generation===current){busy=false;message(e.message,true);render();}}
    };
    document.addEventListener('keydown', e => {
      if (!AnnotationMode.active('gate') || !state || busy || e.repeat || e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
      if (e.key === '[' || e.key === ']') {e.preventDefault();capture(e.key === '['?'gateStart':'gateEnd');}
    });
  });
  return {beginLoad,load,position};
})();

/* One controller timeline shared by playback, manual review and plotted data. */
const Control = (() => {
  const styles = {
    inference: {label: '模型推理', color: '#7c3aed', fill: '#ede9fe'},
    human: {label: '人工介入', color: '#2563eb', fill: '#dbeafe'},
    unknown: {label: '阶段未知', color: '#64748b', fill: '#f1f5f9'},
  };
  let state = null, version = 0;
  const ranges = () => state?.segments || [];
  const nameAt = frame => styles[ranges().find(s => frame >= s.startFrame && frame < s.endFrame)?.phase || 'unknown'].label;

  function reset() {
    version++; state = null;
    $('controlTrack').replaceChildren();
    $('controlSummary').textContent = '正在读取控制阶段…';
    $('controlStatus').className = 'phase-unknown';
    $('controlStatus').textContent = '阶段未知';
  }

  async function load() {
    const generation = ++version, key = meta.key;
    try {
      const result = await api(`/api/control?key=${key}${validOnly() ? '&valid=1' : ''}`);
      if (generation !== version || meta?.key !== key) return;
      state = result;
      $('controlSummary').textContent = Object.entries(styles)
        .filter(([kind]) => kind !== 'unknown' || result.counts[kind])
        .map(([kind, style]) => `${style.label} ${result.counts[kind]} 帧`).join(' · ');
      const track = $('controlTrack'); track.replaceChildren(); appendRanges(track);
      const cursor = document.createElement('span'); cursor.id = 'controlCursor'; cursor.className = 'review-cursor'; track.append(cursor);
      Review.refreshControl(); renderChartBase(); drawChart(); renderRewards(); position(frameAtTime(timelineTime));
    } catch (error) {
      if (generation !== version || meta?.key !== key) return;
      $('controlSummary').textContent = `控制阶段读取失败：${error.message}`;
    }
  }

  function appendRanges(track) {
    const count = state?.totalFrames || 0;
    for (const segment of ranges()) {
      const {startFrame: start, endFrame: end, phase} = segment;
      const button = document.createElement('button');
      button.className = `control-range phase-${phase}`;
      button.style.left = `${start / count * 100}%`;
      button.style.width = `${(end - start) / count * 100}%`;
      button.textContent = `${styles[phase].label} · ${start}–${end - 1}`;
      button.title = `${styles[phase].label}：帧 ${start}–${end - 1}（${(start / meta.fps).toFixed(2)}–${(end / meta.fps).toFixed(2)} 秒），点击跳转`;
      button.setAttribute('aria-label', button.title);
      button.onclick = () => { Review.stopPreview(); pausePlayback(); seekTo(start / meta.fps); };
      track.append(button);
    }
  }

  function position(frame) {
    const phase = ranges().find(s => frame >= s.startFrame && frame < s.endFrame)?.phase || 'unknown';
    $('controlStatus').className = `phase-${phase}`;
    $('controlStatus').textContent = styles[phase].label;
    const cursor = $('controlCursor');
    if (cursor) cursor.style.left = `${Math.min(100, Math.max(0, frame / Math.max(1, state.totalFrames - 1) * 100))}%`;
  }

  function shadeChart(ctx, x, y, width, height, count) {
    if (!count) return;
    for (const s of ranges()) {
      // Each sample owns half the distance to either neighbour.
      const left = count === 1 ? 0 : Math.max(0, s.startFrame - .5) / (count - 1);
      const right = count === 1 ? 1 : Math.min(count - 1, s.endFrame - .5) / (count - 1);
      const span = (right - left) * width;
      ctx.fillStyle = styles[s.phase].fill; ctx.fillRect(x + left * width, y, span, height);
      ctx.fillStyle = styles[s.phase].color; ctx.fillRect(x + left * width, y, span, 3);
      if (span > 74) { ctx.font = '10px sans-serif'; ctx.fillText(styles[s.phase].label, x + left * width + 5, y + 15); }
    }
  }

  return {reset, load, ranges, nameAt, appendRanges, position, shadeChart, counts: () => state?.counts};
})();

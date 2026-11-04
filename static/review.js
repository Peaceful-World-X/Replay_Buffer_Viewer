/* Manual intervals use episode-local frame numbers, inclusive in the editor. */
const Review = (() => {
  let state = null, key = '', version = 0, busy = false, previewEnd = null, previewTimer = null;
  let exportTimer = null, editingId = null;
  const labels = {usable_intervention:'可用介入',usable_motion:'可用动作',recovery:'纠错恢复'};
  const el = id => document.getElementById(id);
  function message(text, error=false){el('reviewMessage').textContent=text;el('reviewMessage').classList.toggle('error',error)}
  function stopPreview(){previewEnd=null;clearInterval(previewTimer);previewTimer=null}
  function frame(){return frameAtTime(playbackVideo()?.readyState>=2?playbackVideo().currentTime:timelineTime)}
  function draftKey(){return `replay-review-draft:${meta.path}:${meta.episodeId}:${meta.key}`}
  function draft(){return {start:el('selectionStart').value,end:el('selectionEnd').value,label:el('selectionLabel').value,note:el('selectionNote').value,outcome:el('selectionOutcome').value,editingId}}
  function persist(){if(!state)return;try{localStorage.setItem(draftKey(),JSON.stringify(draft()))}catch(e){} renderSelection()}
  function clearDraft(){editingId=null;el('selectionOutcome').value='source';['selectionStart','selectionEnd','selectionNote'].forEach(id=>el(id).value='');persist()}
  function selection(){const d=draft();if(d.start===''||d.end==='')throw new Error('请先标记开始和结束。');const start=Number(d.start),end=Number(d.end);if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||end>=meta.totalFrames)throw new Error('请选择至少 2 帧，且开始帧必须早于结束帧。');return {start,end,label:d.label,note:d.note,outcome:d.outcome}}
  function renderSelection(){
    if(!state)return;
    let valid=false;
    try{const s=selection();el('selectionSummary').textContent=`${s.start} → ${s.end} · ${s.end-s.start+1} 帧 / ${s.end-s.start} 条 transition · ${((s.end-s.start+1)/meta.fps).toFixed(2)} 秒`;valid=true}catch(e){el('selectionSummary').textContent=e.message}
    el('saveSelection').textContent=editingId?'保存修改':'保存保留片段';el('saveSelection').disabled=!valid||busy||state.entry.stale;el('previewSelection').disabled=!valid||busy;renderTrack();
  }
  function renderTrack(){
    const track=el('reviewTrack');track.innerHTML='';if(!state)return;
    const bar=(start,end,kind,label)=>{const node=document.createElement('button');node.className=`review-range ${kind}`;node.style.left=`${start/meta.totalFrames*100}%`;node.style.width=`${Math.max(.2,(end-start)/meta.totalFrames*100)}%`;node.title=label;node.setAttribute('aria-label',label);node.onclick=()=>{stopPreview();pausePlayback();seekTo(start/meta.fps)};track.append(node)};
    Control.appendRanges(track);
    state.entry.segments.forEach(s=>bar(s.start_frame,s.end_frame,'kept',`已保留 ${s.start_frame}–${s.end_frame-1}`));
    try{const s=selection();bar(s.start,s.end+1,'draft',`选区 ${s.start}–${s.end}`)}catch(e){}
    const cursor=document.createElement('span');cursor.id='reviewCursor';cursor.className='review-cursor';track.append(cursor);position(timelineTime);
  }
  function syncSidebar(){const item=trajectoryCollection.find(item=>item.path===meta.path&&item.episodeId===meta.episodeId);if(item){item.reviewCount=state.entry.segments.length;item.reviewed=state.entry.reviewed;item.reviewStale=state.entry.stale;renderTrajectoryList()}}
  function render(){
    el('reviewControls').disabled=busy||state.entry.stale;
    el('reviewState').textContent=state.entry.stale?'源数据已改变，标注需核对':state.entry.segments.length?`已保留 ${state.entry.segments.length} 段`:state.entry.reviewed?'已检查 · 未保留片段':'尚未检查';
    renderJumps();
    const list=el('savedSegments');list.innerHTML='';
    state.entry.segments.forEach(s=>{
      const row=document.createElement('div');row.className='saved-segment';
      row.innerHTML=`<div><strong>${s.start_frame} → ${s.end_frame-1} <span>${escapeHtml(labels[s.label])}</span></strong><small>${s.frame_count} 帧 / ${s.frame_count-1} 条 transition · 人工接管 ${s.intervention_frames} 帧${s.note?' · '+escapeHtml(s.note):''}</small></div>`;
      const preview=document.createElement('button');preview.textContent='预览';preview.onclick=()=>previewRange(s.start_frame,s.end_frame-1);
      const restore=document.createElement('button');restore.textContent='修改起止点';restore.onclick=()=>editSegment(s.id);
      const remove=document.createElement('button');remove.textContent='删除';remove.className='delete-segment';remove.onclick=async()=>{if(await mutate(`/api/review/segments/${s.id}`,'DELETE',{},'片段已删除；重新导出后训练目录会同步移除。')){if(editingId===s.id)clearDraft()}};
      row.append(preview,restore,remove);list.append(row);
    });
    if(!state.entry.segments.length)list.textContent='还没有已保存片段。';
    el('downloadReview').href=`/api/review/manifest?key=${key}`;
    renderSelection();syncSidebar();
  }
  function renderJumps(){
    const jumps=el('interventionJumps');jumps.innerHTML='';
    Control.ranges().filter(s=>s.phase==='human').forEach(s=>{const b=document.createElement('button');b.textContent=`${s.startFrame}–${s.endFrame-1}`;b.onclick=()=>{stopPreview();pausePlayback();seekTo(s.startFrame/meta.fps)};jumps.append(b)});
    if(!jumps.children.length){const counts=Control.counts();jumps.textContent=!counts?'正在读取控制阶段…':counts.unknown?'没有已标注的人工介入；灰色区域的控制阶段未知。':'该轨迹全部为模型推理，没有人工介入。'}
  }
  function refreshControl(){if(!state)return;renderTrack();renderJumps();position(timelineTime)}
  function beginLoad(){version++;state=null;key='';editingId=null;stopPreview();el('reviewControls').disabled=true;el('reviewPanel').classList.add('hidden')}
  async function load(){
    if(meta.sourceType!=='lerobot')return;
    const generation=version,currentKey=meta.key;key=currentKey;
    el('reviewPanel').classList.remove('hidden');message('正在读取标注…');
    try{
      const result=await api(`/api/review?key=${currentKey}`);if(version!==generation||meta.key!==currentKey)return;state=result;
      let d={};try{d=JSON.parse(localStorage.getItem(draftKey())||'{}')}catch(e){}
      el('selectionStart').value=d.start??'';el('selectionEnd').value=d.end??'';el('selectionNote').value=d.note??'';el('selectionLabel').value=d.label||'usable_intervention';el('selectionOutcome').value=d.outcome||'source';editingId=state.entry.segments.some(item=>item.id===d.editingId)?d.editingId:null;
      ['selectionStart','selectionEnd'].forEach(id=>el(id).max=meta.totalFrames-1);
      meta.trainingExists=result.trainingExists;updateNavigation(meta);render();message(state.entry.stale?'源数据已改变，旧标注已锁定，请核对源文件。':`标注保存在 ${state.annotationPath}`,state.entry.stale);
    }catch(e){if(version===generation)message(e.message,true)}
  }
  async function mutate(url,method,body,success){
    if(!state||busy)return false;
    const generation=version,currentKey=key;busy=true;render();
    try{
      const result=await api(`${url}?key=${currentKey}`,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,revision:state.entry.revision})});
      if(generation!==version)return false;state=result;message(success);return true;
    }catch(e){if(generation===version)message(e.message,true);return false}
    finally{busy=false;if(state)render()}
  }
  function capture(which){if(!state||busy||state.entry.stale)return;stopPreview();const current=frame();pausePlayback();seekTo(current/meta.fps);el(which==='start'?'selectionStart':'selectionEnd').value=current;if(which==='start'&&(el('selectionEnd').value===''||Number(el('selectionEnd').value)<=current))el('selectionEnd').value=meta.totalFrames-1;persist();message(`已标记${which==='start'?'开始':'结束'}：Episode ${meta.episodeId}，帧 ${current}。`)}
  async function previewRange(start,end){
    stopPreview();pausePlayback();seekTo(start/meta.fps);previewEnd=end;
    if(start===end){message('单帧选区已定位。');return}
    await togglePlay();if(!playing){stopPreview();return}
    previewTimer=setInterval(()=>{if(previewEnd===null||!meta)return;const time=playbackVideo()?.currentTime??timelineTime;if(time>=(previewEnd+1)/meta.fps||!playing){const target=previewEnd;stopPreview();pausePlayback();seekTo(target/meta.fps)}},25);
  }
  function editSegment(id){if(!state)return;const s=state.entry.segments.find(item=>item.id===id);if(!s)return;editingId=id;el('selectionStart').value=s.start_frame;el('selectionEnd').value=s.end_frame-1;el('selectionLabel').value=s.label;el('selectionOutcome').value=s.outcome||'source';el('selectionNote').value=s.note;persist();stopPreview();pausePlayback();seekTo(s.start_frame/meta.fps);message('调整开始或结束帧，点击保存修改，再导出覆盖训练轨迹。')}
  function position(time){if(!state)return;const f=frameAtTime(time);el('reviewPosition').textContent=`Episode ${meta.episodeId} · 帧 ${f} · ${Control.nameAt(f)}`;const cursor=el('reviewCursor');if(cursor)cursor.style.left=`${f/Math.max(1,meta.totalFrames-1)*100}%`}
  async function exportSaved(){
    if(!state||el('exportReview').disabled)return;
    el('exportReview').disabled=true;el('exportMessage').textContent='正在生成训练轨迹，完成后更新固定子目录…';
    try{const result=await api(`/api/review/export?key=${key}`,{method:'POST'});sessionStorage.setItem('reviewExportJob',result.jobId);pollExport(result.jobId)}
    catch(e){el('exportMessage').textContent=e.message;el('exportReview').disabled=false}
  }
  async function pollExport(id){
    clearTimeout(exportTimer);
    try{const job=await api(`/api/review/export/${id}`);el('exportMessage').textContent=job.status==='complete'?`导出完成：${job.result.episodes} 个片段，${job.result.frames} 帧 / ${job.result.transitions} 条 transition。目录：${job.result.directory}`:job.message;
      if(job.status==='running'){el('exportReview').disabled=true;exportTimer=setTimeout(()=>pollExport(id),1000)}else{el('exportReview').disabled=false;sessionStorage.removeItem('reviewExportJob');if(job.status==='complete'&&meta){if(meta.sourceType==='lerobot'&&meta.path===job.source){meta.trainingExists=true;updateNavigation(meta)}else if(meta.sourceType==='training_replay'&&meta.trainingDirectory===job.result.directory){await openDataset(job.result.directory)}}}
    }catch(e){el('exportMessage').textContent=e.message;el('exportReview').disabled=false;sessionStorage.removeItem('reviewExportJob')}
  }
  document.addEventListener('DOMContentLoaded',()=>{
    el('markStart').onclick=()=>capture('start');el('markEnd').onclick=()=>capture('end');
    ['selectionStart','selectionEnd','selectionLabel','selectionNote','selectionOutcome'].forEach(id=>el(id).addEventListener('input',persist));
    el('clearSelection').onclick=()=>{stopPreview();clearDraft();message('选区已清空，已保存片段保留。')};
    el('previewSelection').onclick=()=>{try{const s=selection();previewRange(s.start,s.end)}catch(e){message(e.message,true)}};
    el('saveSelection').onclick=async()=>{try{const s=selection();if(await mutate('/api/review/segments'+(editingId?'/'+editingId:''),editingId?'PUT':'POST',{startFrame:s.start,endFrameInclusive:s.end,label:s.label,note:s.note,outcome:s.outcome},'片段已保存，导出后训练目录会同步更新。'))clearDraft()}catch(e){message(e.message,true)}};
    el('completeReview').onclick=()=>mutate('/api/review/complete','POST',{},'此 episode 已标记检查完成。');
    el('exportReview').onclick=exportSaved;
    document.addEventListener('keydown',e=>{if(!state||busy||state.entry.stale||e.repeat||e.ctrlKey||e.metaKey||e.altKey||(/TEXTAREA|SELECT/.test(e.target.tagName)||(e.target.tagName==='INPUT'&&e.target.type!=='range'))||e.target.isContentEditable)return;
      if(['i','o',' ','ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();if(e.key==='i')capture('start');else if(e.key==='o')capture('end');else if(e.key===' ')togglePlay();else{stopPreview();pausePlayback();seekTo((frame()+(e.key==='ArrowLeft'?-1:1))/meta.fps)}}});
    const job=sessionStorage.getItem('reviewExportJob');if(job)pollExport(job);
  });
  return {beginLoad,load,position,editSegment,refreshControl,stopPreview};
})();

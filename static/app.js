const $ = (id) => document.getElementById(id);
const videos = [$('videoMain'), $('videoLeft'), $('videoRight')];
let meta = null, chartValues = [], playing = false, seeking = false, frameRequest = 0, frameRequestVersion = 0;
let hiddenComponents = new Set(), chartBase = document.createElement('canvas'), chartRange = {min:0,max:1};
let rewardValues = [], rewardSourceIndices = [];
let activeEpisodeId = null;
let trajectoryCollection = [], activeTrajectoryPath = '', trajectoryLoadVersion = 0;
let timelineTime = 0, playbackTimer = null, playbackVersion = 0, playbackError = false;
const cameraNames = ['main', 'left', 'right'];
const AnnotationMode = (() => {
  let mode = 'gate', available = false;
  try { if (localStorage.getItem('replay-annotation-mode') === 'review') mode = 'review'; } catch (_) {}
  function name() { return mode === 'gate' ? 'Actor 起止标定' : '片段裁剪'; }
  function render() {
    document.body.dataset.annotationMode = mode;
    document.body.dataset.annotationAvailable = String(available);
    $('annotationModeBar').classList.toggle('hidden', !available);
    $('annotationModeName').textContent = name();
    $('annotationModeHelp').textContent = mode === 'gate'
      ? '设置 actor 的开启帧和关闭帧，生成完整轨迹的 0/1 标签。快捷键：[ 开启，] 关闭。'
      : '选择需要保留的动作片段并导出裁剪数据。快捷键：I 开始，O 结束。';
    $('annotationModeToggle').textContent = mode === 'gate' ? '切换到片段裁剪' : '切换到 Actor 起止标定';
  }
  function select(next) {
    mode = next === 'review' ? 'review' : 'gate';
    Review.stopPreview();
    try { localStorage.setItem('replay-annotation-mode', mode); } catch (_) {}
    render();
    if (navigationContext) updateNavigation(navigationContext);
  }
  $('annotationModeToggle').onclick = () => select(mode === 'gate' ? 'review' : 'gate');
  render();
  return {name, select, active: value => available && mode === value,
    setAvailable(value) { available = value; render(); }};
})();
function activeVideos(){return videos.filter((v,i)=>meta?.cameras.includes(cameraNames[i]))}
function playbackVideo(){return activeVideos()[0]}
function pausePlayback(){playbackVersion++;playing=false;clearInterval(playbackTimer);playbackTimer=null;videos.forEach(v=>v.pause());$('playIcon').innerHTML='&#9654;'}

function validOnly(){return $('validOnly').checked}
function activeFrames(){return validOnly()?meta.validFrames:meta.totalFrames}
function activeDuration(){return activeFrames()/meta.fps}
// Browser media clocks round to microseconds; allow that rounding at frame edges.
function frameAtTime(time){return Math.min(activeFrames()-1,Math.max(0,Math.floor(time*meta.fps+1e-4)))}
function stepFrame(delta){if(!meta)return;pausePlayback();seekTo((frameAtTime(timelineTime)+delta)/meta.fps)}
function validQuery(){return (validOnly()?'&valid=1':'')+`&key=${meta.key}`}

function formatBytes(n){const u=['B','KiB','MiB','GiB'];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return `${n.toFixed(i?1:0)} ${u[i]}`}
function formatTime(s){if(!Number.isFinite(s))s=0;const m=Math.floor(s/60),r=s-m*60;return `${String(m).padStart(2,'0')}:${r.toFixed(3).padStart(6,'0')}`}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function outcomeDisplay(item){return item.isSuccess===true?{name:'成功',className:'outcome-success'}:item.isSuccess===false?{name:'失败',className:'outcome-failure'}:{name:'未知',className:'outcome-unknown'}}
function updateOutcome(item){const result=outcomeDisplay(item),badge=$('trajectoryOutcome');badge.textContent=result.name;badge.className='outcome-badge '+result.className;badge.title=item.outcomeSource||'未记录明确结果'}

async function api(url, options){const response=await fetch(url,options),type=response.headers.get('content-type')||'';if(!type.includes('application/json'))throw new Error(`Server returned ${response.status} ${type||'non-JSON response'}; restart run.sh and refresh this page`);const body=await response.json();if(!response.ok)throw new Error(body.error||response.statusText);return body}

let navigationContext=null;
function updateNavigation(context){navigationContext=context;const original=context.sourceType==='lerobot',training=context.sourceType==='training_replay';$('datasetNav').classList.toggle('hidden',!original&&!training);if(!original&&!training)return;$('datasetMode').textContent=original?`原始数据 · ${AnnotationMode.name()}`:'训练轨迹 · 导出结果预览';const origin=training&&context.sourceRange?` · 原始 Episode ${context.sourceEpisodeId} · 帧 ${context.sourceRange[0]}–${context.sourceRange[1]-1}`:'';$('datasetLocation').textContent=(training?context.trainingDirectory:context.sourceDirectory||context.directory)+origin;$('openTraining').classList.toggle('hidden',!original);$('openTraining').disabled=!context.trainingExists;$('returnOriginal').classList.toggle('hidden',!training)}
async function openDataset(path,episodeId=null,segmentId=null){$('pathInput').value=path;await openInput({episodeId,segmentId})}
async function openInput(options={}){
  const path=$('pathInput').value.trim();if(!path)return;
  if(path.toLowerCase().endsWith('.pt')){hideTrajectorySidebar();await loadTrajectory(path);return}
  $('loadBtn').disabled=true;$('loadBtn').textContent='正在读取目录…';$('loadError').textContent='';
  try{
    const collection=await api('/api/browse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});trajectoryCollection=collection.trajectories;$('interventionFilter').value='all';$('outcomeFilter').value='all';
    history.replaceState(null,'',`?file=${encodeURIComponent(path)}${options.episodeId!==null&&options.episodeId!==undefined?'&episode='+options.episodeId:''}`);
    $('emptyTraining').classList.toggle('hidden',trajectoryCollection.length>0);
    if(!trajectoryCollection.length){AnnotationMode.setAvailable(false);Review.beginLoad();Gate.beginLoad();pausePlayback();meta=null;hideTrajectorySidebar();$('workspace').classList.add('hidden');updateNavigation(collection);$('statusBadge').innerHTML='<span></span>暂无保留轨迹';return}
    showTrajectorySidebar(collection);const selected=trajectoryCollection.find(item=>item.episodeId===options.episodeId)||trajectoryCollection[0];await loadTrajectory(selected.path,true,selected.episodeId);
    if(options.segmentId){AnnotationMode.select('review');Review.editSegment(options.segmentId);}
  }
  catch(error){$('loadError').textContent=error.message;$('statusBadge').className='status idle';$('statusBadge').innerHTML='<span></span>载入失败'}
  finally{$('loadBtn').disabled=false;$('loadBtn').textContent='打开数据'}
}

async function loadTrajectory(path,fromCollection=false,episodeId=null){
  if(!path)return;const loadVersion=++trajectoryLoadVersion;frameRequestVersion++;clearTimeout(frameRequest);AnnotationMode.setAvailable(false);Review.beginLoad();Gate.beginLoad();Control.reset();
  $('loadBtn').disabled=true;$('loadBtn').textContent='正在解析…';$('loadError').textContent='';
  $('trajectoryOutcome').textContent='载入中';$('trajectoryOutcome').className='outcome-badge outcome-unknown';$('trajectoryOutcome').title='';
  try{
    pausePlayback();videos.forEach(v=>{v.removeAttribute('src');v.load()});
    const loaded=await api('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,episodeId})});if(loadVersion!==trajectoryLoadVersion)return;meta=loaded;AnnotationMode.setAvailable(meta.sourceType==='lerobot');activeEpisodeId=meta.episodeId??null;
    activeTrajectoryPath=path;frameRequestVersion++;markActiveTrajectory();updateNavigation(meta);$('emptyTraining').classList.add('hidden');
    updateOutcome(meta);
    $('workspace').classList.remove('hidden');$('statusBadge').className='status ready';$('statusBadge').innerHTML='<span></span>数据已载入';
    $('fileName').textContent=meta.filename;$('validFrames').textContent=`${meta.validFrames} / ${meta.totalFrames}`;$('chunks').textContent=`${meta.chunks} × ${meta.stepsPerChunk}`;$('fileSize').textContent=formatBytes(meta.fileBytes);$('weightsId').textContent=meta.modelWeightsId;
    $('chunksLabel').textContent=meta.sourceType==='lerobot'?'数据帧':'Chunks';$('weightsLabel').textContent=meta.sourceType==='lerobot'?'数据格式':'Weights ID';if(meta.sourceType==='lerobot')$('chunks').textContent=meta.totalFrames;if(meta.sourceType==='training_replay'){$('chunksLabel').textContent='Transitions';$('weightsLabel').textContent='导出格式'}
    $('validOnly').disabled=!meta.hasValidMask;$('validOnly').checked=meta.hasValidMask;
    refreshMedia();
    buildSeries();renderTree(meta.tree,$('dataTree'));await Promise.all([loadDefaultSeries(),loadRewards(),Control.load(),Review.load(),Gate.load()]);if(loadVersion!==trajectoryLoadVersion)return;seekTo(0);window.scrollTo({top:0,behavior:'smooth'});
  }catch(error){updateOutcome(meta||{});$('loadError').textContent=error.message;$('statusBadge').className='status idle';$('statusBadge').innerHTML='<span></span>载入失败'}
  finally{$('loadBtn').disabled=false;$('loadBtn').textContent='打开数据'}
}

function showTrajectorySidebar(collection){const sidebar=$('trajectorySidebar'),shell=$('collectionShell');sidebar.classList.remove('hidden');$('sidebarOpen').classList.add('hidden');shell.classList.add('with-sidebar');const metadata=collection.metadata||{};$('collectionCount').textContent=`${trajectoryCollection.length} ${collection.sourceType==='lerobot'?'episodes':'trajectories'} · ${metadata.total_samples??0} ${collection.sourceType==='lerobot'?'frames':'samples'}`;renderTrajectoryList()}
function hideTrajectorySidebar(){$('trajectorySidebar').classList.add('hidden');$('sidebarOpen').classList.add('hidden');$('collectionShell').classList.remove('with-sidebar');trajectoryCollection=[]}
function collapseTrajectorySidebar(){$('trajectorySidebar').classList.add('hidden');$('sidebarOpen').classList.remove('hidden');$('collectionShell').classList.remove('with-sidebar')}
function expandTrajectorySidebar(){$('trajectorySidebar').classList.remove('hidden');$('sidebarOpen').classList.add('hidden');$('collectionShell').classList.add('with-sidebar')}
function filteredTrajectories(){
  const query=$('trajectorySearch').value.trim().toLowerCase(),humanOnly=$('interventionFilter').value==='human',outcome=$('outcomeFilter').value;
  return trajectoryCollection.filter(item=>(!humanOnly||item.hasIntervention)
    &&(outcome==='all'||(outcome==='success'&&item.isSuccess===true)||(outcome==='failure'&&item.isSuccess===false))
    &&(!query||String(item.id).includes(query)||item.modelWeightsId.toLowerCase().includes(query)||item.filename.toLowerCase().includes(query)));
}
function renderTrajectoryList(){
  const list=$('trajectoryList'),items=filteredTrajectories(),humanCount=trajectoryCollection.filter(item=>item.hasIntervention).length;
  list.innerHTML='';$('interventionFilter').options[1].textContent=`只看含人工介入（${humanCount}）`;
  $('outcomeFilter').options[1].textContent=`只看成功（${trajectoryCollection.filter(item=>item.isSuccess===true).length}）`;
  $('outcomeFilter').options[2].textContent=`只看失败（${trajectoryCollection.filter(item=>item.isSuccess===false).length}）`;
  const successes=items.filter(item=>item.isSuccess===true).length,failures=items.filter(item=>item.isSuccess===false).length,unknown=items.length-successes-failures;
  $('trajectoryFilterCount').textContent=`显示 ${items.length} / ${trajectoryCollection.length} 条 · 含介入 ${humanCount} 条\n当前列表：成功 ${successes} · 失败 ${failures} · 未知 ${unknown}`;
  items.forEach(item=>{
    const outcome=outcomeDisplay(item);
    const button=document.createElement('button');
    button.className='trajectory-item'+(item.path===activeTrajectoryPath&&(item.episodeId??null)===activeEpisodeId?' active':'')+(item.isSuccess===false?' failed':item.hasIntervention?' intervened':'');
    button.dataset.path=item.path;button.dataset.episode=item.episodeId??'';
    button.innerHTML=`<strong>${item.episodeId!==undefined?'Episode':'Trajectory'} ${item.id}</strong><div class="trajectory-badges"><span class="outcome-badge ${outcome.className}" title="${escapeHtml(item.outcomeSource||'未记录明确结果')}">${outcome.name}</span>${item.hasIntervention?'<span class="intervention-badge">含人工介入</span>':''}</div><small>${escapeHtml(item.modelWeightsId)}</small><span class="item-meta"><span>${item.numSamples} samples</span><span>${formatBytes(item.fileBytes)}</span></span>${item.episodeId!==undefined?`<small class="gate-sidebar-badge">门控：${item.gateReviewed?'已标注':'未标注'}</small><small class="review-badge">${item.reviewStale?'源数据已变化':item.reviewCount?'已保留 '+item.reviewCount+' 段':item.reviewed?'已检查 · 无保留':'待检查'}</small>`:''}`;
    button.onclick=()=>loadTrajectory(item.path,true,item.episodeId);list.append(button);
  });
  if(!items.length){const empty=document.createElement('p');empty.className='trajectory-empty';empty.textContent='没有匹配的轨迹，可清空搜索，或将筛选改为“全部轨迹”和“全部结果”。';list.append(empty)}
}
async function applyTrajectoryFilters(){
  renderTrajectoryList();const items=filteredTrajectories();
  if(items.length&&!items.some(item=>item.path===activeTrajectoryPath&&(item.episodeId??null)===activeEpisodeId))await loadTrajectory(items[0].path,true,items[0].episodeId);
}
function markActiveTrajectory(){document.querySelectorAll('.trajectory-item').forEach(item=>item.classList.toggle('active',item.dataset.path===activeTrajectoryPath&&item.dataset.episode===String(activeEpisodeId??'')))}

function buildSeries(){const sel=$('seriesSelect');sel.innerHTML='';meta.series.forEach(item=>{const option=document.createElement('option');option.value=item.path;option.textContent=`${item.path}  [${item.sourceShape.join('×')}]`;sel.append(option)});const preferred=['forward_inputs.substep_states','curr_obs.states','actions'].find(path=>meta.series.some(x=>x.path===path));if(preferred)sel.value=preferred}
function curveColor(index,total){return `hsl(${Math.round(index*360/Math.max(total,1))} 68% ${index%2?43:36}%)`}
function componentLabel(index,shape){if(!shape.length)return 'value';if(shape.length===1)return `dim ${index}`;const coords=[];let n=index;for(let i=shape.length-1;i>=0;i--){coords.unshift(n%shape[i]);n=Math.floor(n/shape[i])}return `[${coords.join(', ')}]`}
function buildLegend(shape,components){const legend=$('legend');legend.innerHTML='';for(let i=0;i<components;i++){const item=document.createElement('button');item.className='legend-item';item.dataset.component=i;item.innerHTML=`<span class="legend-swatch" style="background:${curveColor(i,components)}"></span>${componentLabel(i,shape)}`;item.onclick=()=>{hiddenComponents.has(i)?hiddenComponents.delete(i):hiddenComponents.add(i);item.classList.toggle('off',hiddenComponents.has(i));renderChartBase();drawChart()};legend.append(item)}}
async function loadDefaultSeries(){const path=$('seriesSelect').value,requestKey=meta.key;try{const result=await api(`/api/series?path=${encodeURIComponent(path)}${validQuery()}`);if(meta.key!==requestKey||$('seriesSelect').value!==path)return;chartValues=result.values;hiddenComponents.clear();buildLegend(result.componentShape,result.components);renderChartBase();drawChart()}catch(e){$('legend').textContent=e.message}}
async function loadRewards(){
  const hasRewards=meta.series.some(item=>item.path==='rewards');$('rewardStrip').classList.toggle('hidden',!hasRewards);
  if(!hasRewards){rewardValues=[];rewardSourceIndices=[];return}
  const requestKey=meta.key;
  try{
    const result=await api(`/api/series?path=rewards${validQuery()}`);if(meta?.key!==requestKey)return;
    rewardValues=result.values.map(row=>Number(row[0]));rewardSourceIndices=result.sourceIndices;
    $('rewardEvents').textContent=rewardValues.filter(value=>value!==0).length;$('rewardTotal').textContent=rewardValues.reduce((sum,value)=>sum+value,0).toFixed(4);renderRewards();updateReward(0);
  }catch(e){if(meta?.key!==requestKey)return;rewardValues=[];$('rewardEvents').textContent='-';$('rewardTotal').textContent='-';renderRewards()}
}

function refreshMedia(){
  if(!meta)return;
  pausePlayback();timelineTime=0;playbackError=false;
  const duration=activeDuration();$('duration').textContent=`${duration.toFixed(2)} s`;$('timeline').max=Math.max(0,(activeFrames()-1)/meta.fps);$('timeline').value=0;$('timeReadout').textContent=`${formatTime(0)} / ${formatTime(duration)}`;
  const available=activeVideos(),hasFrames=activeFrames()>0;
  $('cameraGrid').classList.toggle('hidden',!available.length);$('cameraGrid').style.setProperty('--camera-count',Math.max(1,available.length));
  $('mediaDescription').textContent=meta.videoSource==='external'?'按原始视频帧索引同步播放':meta.videoSource==='embedded'?'前置与双腕相机':'无相机视频 · 可播放和逐帧查看数值数据';
  const message=meta.mediaMessage+(meta.fpsAssumed?' 数值回放速度暂按 30 帧/秒。':'');$('mediaMessage').textContent=message;$('mediaMessage').classList.toggle('hidden',!message);
  ['playBtn','backBtn','forwardBtn','timeline'].forEach(id=>$(id).disabled=!hasFrames);
  const stamp=Date.now(),mode=validQuery();videos.forEach((v,i)=>{const enabled=available.includes(v)&&hasFrames;v.closest('.camera').classList.toggle('hidden',!enabled);if(enabled){v.src=`/api/video/${cameraNames[i]}.mp4?v=${stamp}${mode}`;v.playbackRate=Number($('speedSelect').value)}else v.removeAttribute('src');v.load()});
}

function syncVideos(source){if(seeking||!activeVideos().includes(source))return;const t=source.currentTime;activeVideos().forEach(v=>{if(v!==source&&Math.abs(v.currentTime-t)>.08)v.currentTime=t})}
function seekTo(time){if(!meta||!activeFrames())return;time=Math.max(0,Math.min((activeFrames()-1)/meta.fps,time));seeking=true;activeVideos().forEach(v=>{v.currentTime=time+1e-6});seeking=false;updateTimeline(time)}
async function togglePlay(){
  if(!meta||!activeFrames())return;
  if(playing){pausePlayback();return}
  if(timelineTime>=(activeFrames()-1)/meta.fps)seekTo(0);
  playing=true;const requestVersion=++playbackVersion,requestKey=meta.key;$('playIcon').innerHTML='&#10074;&#10074;';
  const available=activeVideos();
  if(available.length){
    try{
      await Promise.all(available.map(v=>v.play()));
      if(requestVersion!==playbackVersion||meta?.key!==requestKey)return;
      if(playbackError){playbackError=false;const message=meta.mediaMessage+(meta.fpsAssumed?' 数值回放速度暂按 30 帧/秒。':'');$('mediaMessage').textContent=message;$('mediaMessage').classList.toggle('hidden',!message)}
    }catch(e){
      // A pause, seek-and-mark, or new trajectory can cancel a pending play request.
      if(requestVersion!==playbackVersion||meta?.key!==requestKey)return;
      pausePlayback();if(e.name==='AbortError')return;
      playbackError=true;$('mediaMessage').textContent='视频播放失败：'+e.message;$('mediaMessage').classList.remove('hidden');
    }
    return;
  }
  let last=performance.now();playbackTimer=setInterval(()=>{const now=performance.now(),next=timelineTime+(now-last)/1000*Number($('speedSelect').value);last=now;seekTo(next);if(next>=(activeFrames()-1)/meta.fps)pausePlayback()},100);
}
function updateTimeline(time){if(!meta||!activeFrames())return;timelineTime=time;if(!seeking)$('timeline').value=time;const frame=frameAtTime(time);$('timeReadout').textContent=`${formatTime(time)} / ${formatTime(activeDuration())}`;$('frameReadout').textContent=`Frame ${frame} / ${activeFrames()-1}`;if(meta.sourceType==='lerobot')$('chunkReadout').textContent=`Episode ${meta.episodeId} · 帧 ${frame}`;else if(!validOnly())$('chunkReadout').textContent='Chunk '+Math.floor(frame/meta.stepsPerChunk)+' · Step '+frame%meta.stepsPerChunk;drawChart(frame);updateReward(frame);Review.position(time);Gate.position(time);Control.position(frame);clearTimeout(frameRequest);const version=++frameRequestVersion;frameRequest=setTimeout(()=>loadFrame(frame,version),80)}
async function loadFrame(frame,version){try{const result=await api(`/api/frame?index=${frame}${validQuery()}`);if(version!==frameRequestVersion)return;const position=meta.sourceType==='lerobot'?`Episode ${meta.episodeId} · 帧 ${result.sourceIndex}`:`Chunk ${result.chunk} · Step ${result.step}`;$('currentLocation').textContent=position;$('chunkReadout').textContent=position;$('frameValues').innerHTML=Object.entries(result.values).map(([k,v])=>`<div class="value-row"><label>${escapeHtml(k)}</label><code>${escapeHtml(Array.isArray(v)?v.map(x=>typeof x==='number'?Number(x).toFixed(4):x).join(', '):String(v))}</code></div>`).join('')}catch(e){}}

function renderRewards(){const canvas=$('rewardCanvas'),box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,box.width),h=Math.max(1,box.height);canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);const mid=h/2,bar=Math.max(1,w/Math.max(1,rewardValues.length));Control.ranges().filter(s=>s.phase==='human').forEach(s=>{if(rewardValues.length){c.fillStyle='#dbeafe';c.fillRect(s.startFrame/rewardValues.length*w,0,(s.endFrame-s.startFrame)/rewardValues.length*w,h)}});c.strokeStyle='#dfe4e8';c.lineWidth=1;c.beginPath();c.moveTo(0,mid);c.lineTo(w,mid);c.stroke();if(!rewardValues.length)return;const maxAbs=Math.max(1e-9,...rewardValues.map(Math.abs));rewardValues.forEach((value,index)=>{const x=index/rewardValues.length*w;if(value===0){c.fillStyle='#cbd3da';c.fillRect(x,mid-1,Math.max(1,bar),2);return}const height=Math.max(5,Math.abs(value)/maxAbs*(h/2-8));c.fillStyle=value>0?'#087f5b':'#c92a2a';c.fillRect(x,value>0?mid-height:mid,Math.max(2,bar),height)});c.fillStyle='#7b8793';c.font='9px sans-serif';c.fillText('+',5,11);c.fillText('0',5,mid-4);c.fillText('−',5,h-5)}
function updateReward(frame){if(!rewardValues.length)return;frame=Math.max(0,Math.min(rewardValues.length-1,frame));const value=rewardValues[frame];$('rewardCurrent').textContent=value.toFixed(4);$('rewardCurrent').style.color=value<0?'#c92a2a':value>0?'#087f5b':'#667482';$('rewardCursor').style.left=`${frame/Math.max(1,rewardValues.length-1)*100}%`}
function rewardFrameFromEvent(event){const box=$('rewardTrack').getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(event.clientX-box.left)/box.width));return Math.round(ratio*Math.max(0,rewardValues.length-1))}
function showRewardTooltip(event){if(!rewardValues.length)return;const frame=rewardFrameFromEvent(event),source=rewardSourceIndices[frame]??frame,chunk=Math.floor(source/meta.stepsPerChunk),step=source%meta.stepsPerChunk,value=rewardValues[frame],tip=$('rewardTooltip'),box=$('rewardTrack').getBoundingClientRect();tip.innerHTML=`<b>Reward ${value.toFixed(6)}</b><br>Frame ${frame} · Chunk ${chunk} · Step ${step}<br>${formatTime(frame/meta.fps)}`;tip.style.display='block';const desired=event.clientX-box.left+10;tip.style.left=`${Math.min(desired,box.width-tip.offsetWidth-6)}px`}

function renderChartBase(){const canvas=$('chart'),box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,box.width),h=Math.max(1,box.height),p={l:48,r:14,t:13,b:27};chartBase.width=w*dpr;chartBase.height=h*dpr;const c=chartBase.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);if(!chartValues.length)return;const componentCount=chartValues[0]?.length||0,visible=[];for(let j=0;j<componentCount;j++)if(!hiddenComponents.has(j))for(const row of chartValues){const v=Number(row[j]);if(Number.isFinite(v))visible.push(v)}let min=visible.length?Math.min(...visible):0,max=visible.length?Math.max(...visible):1;if(min===max){min-=1;max+=1}chartRange={min,max};Control.shadeChart(c,p.l,p.t,w-p.l-p.r,h-p.t-p.b,chartValues.length);const x=i=>p.l+i/Math.max(1,chartValues.length-1)*(w-p.l-p.r),y=v=>p.t+(max-v)/(max-min)*(h-p.t-p.b);c.strokeStyle='#e3e8ed';c.lineWidth=1;c.fillStyle='#74808c';c.font='10px sans-serif';for(let i=0;i<=4;i++){const yy=p.t+i/4*(h-p.t-p.b),value=max-i/4*(max-min);c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();c.fillText(value.toFixed(2),4,yy+3)}for(let j=0;j<componentCount;j++){if(hiddenComponents.has(j))continue;c.strokeStyle=curveColor(j,componentCount);c.lineWidth=1.25;c.beginPath();chartValues.forEach((row,i)=>{const xx=x(i),yy=y(Number(row[j]));if(i)c.lineTo(xx,yy);else c.moveTo(xx,yy)});c.stroke()}c.fillStyle='#74808c';c.fillText('0',p.l,h-8);c.fillText(String(chartValues.length-1),w-p.r-24,h-8)}
function drawChart(frame){const canvas=$('chart'),box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,box.width),h=Math.max(1,box.height),p={l:48,r:14,t:13,b:27};if(canvas.width!==w*dpr||canvas.height!==h*dpr){renderChartBase();canvas.width=w*dpr;canvas.height=h*dpr}const c=canvas.getContext('2d');c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,canvas.width,canvas.height);c.drawImage(chartBase,0,0);if(!chartValues.length)return;const idx=frame===undefined?Math.floor(timelineTime*(meta?.fps||30)):frame;if(idx>=0&&idx<chartValues.length){const xx=(p.l+idx/Math.max(1,chartValues.length-1)*(w-p.l-p.r))*dpr;c.strokeStyle='#e8590c';c.lineWidth=dpr;c.beginPath();c.moveTo(xx,p.t*dpr);c.lineTo(xx,(h-p.b)*dpr);c.stroke()}}

function renderTree(root,container){container.innerHTML='';const make=(key,node,depth=0)=>{const wrap=document.createElement('div');wrap.className='tree-node';const row=document.createElement('div');row.className='tree-row';const children=node.children;row.innerHTML=`${children?'<button class="tree-toggle">▾</button>':'<span class="tree-toggle"></span>'}<span class="tree-key">${escapeHtml(key)}</span><span class="tree-type ${node.type==='h264'?'h264':''}">${escapeHtml(node.type)}</span><span class="tree-meta">${node.shape?'['+node.shape.join(' × ')+'] ':''}${node.dtype?escapeHtml(node.dtype)+' ':''}${node.bytes!==undefined?formatBytes(node.bytes):''}${node.value!==undefined?'= '+escapeHtml(node.value):''}</span>`;wrap.append(row);if(children){const childWrap=document.createElement('div');childWrap.className='tree-children';Object.entries(children).forEach(([k,v])=>childWrap.append(make(k,v,depth+1)));wrap.append(childWrap);row.querySelector('button').onclick=()=>{childWrap.classList.toggle('collapsed');row.querySelector('button').textContent=childWrap.classList.contains('collapsed')?'▸':'▾'}}return wrap};container.append(make('trajectory',root))}

$('loadBtn').onclick=openInput;$('pathInput').onkeydown=e=>{if(e.key==='Enter')openInput()};$('sidebarToggle').onclick=collapseTrajectorySidebar;$('sidebarOpen').onclick=expandTrajectorySidebar;$('trajectorySearch').oninput=renderTrajectoryList;$('interventionFilter').onchange=applyTrajectoryFilters;$('outcomeFilter').onchange=applyTrajectoryFilters;$('playBtn').onclick=togglePlay;$('backBtn').onclick=()=>stepFrame(-1);$('forwardBtn').onclick=()=>stepFrame(1);$('timeline').oninput=e=>seekTo(Number(e.target.value));$('validOnly').onchange=async()=>{refreshMedia();await Promise.all([loadDefaultSeries(),loadRewards(),Control.load()]);seekTo(0)};$('speedSelect').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));$('seriesSelect').onchange=loadDefaultSeries;videos.forEach(video=>{video.ontimeupdate=()=>{if(video===playbackVideo())updateTimeline(video.currentTime)};video.onended=()=>{if(video===playbackVideo())pausePlayback()};video.addEventListener('seeking',()=>syncVideos(video));video.onerror=()=>{if(!video.getAttribute('src'))return;$('mediaMessage').textContent='相机视频加载失败，可继续查看数值数据。请检查源视频文件和服务端日志。';$('mediaMessage').classList.remove('hidden')}});$('rewardTrack').onmousemove=showRewardTooltip;$('rewardTrack').onmouseleave=()=>{$('rewardTooltip').style.display='none'};$('rewardTrack').onclick=e=>seekTo(rewardFrameFromEvent(e)/(meta?.fps||30));window.addEventListener('resize',()=>{renderChartBase();drawChart();renderRewards()});$('treeSearch').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.tree-node').forEach(n=>{const own=n.firstElementChild?.querySelector('.tree-key')?.textContent.toLowerCase()||'';n.style.display=!q||own.includes(q)||n.textContent.toLowerCase().includes(q)?'':'none'})};

$('openTraining').onclick=()=>openDataset(navigationContext.trainingDirectory);$('returnOriginal').onclick=()=>openDataset(navigationContext.sourceDirectory,navigationContext.sourceEpisodeId??null,meta?.segmentId??null);
const params=new URLSearchParams(location.search);if(params.get('file')){$('pathInput').value=params.get('file');openInput({episodeId:params.has('episode')?Number(params.get('episode')):null,segmentId:params.get('segment')})}

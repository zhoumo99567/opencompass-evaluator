// Capability taxonomy, persistent selection across filters, and curated bundles.
const datasetPicker={category:'all',scope:'all',language:'all',availability:'all',query:'',page:1,size:8,openInfo:new Set(),activePreset:null,optionsOpen:false};
const benchmarkCategories=()=>state.benchmarkMetadata?.categories||[];
const benchmarkPresets=()=>state.benchmarkMetadata?.presets||[];
const benchmarkBusy=()=>state.starting||state.evalStatus?.status==='running';
const languageName=value=>({zh:'中文',en:'英文',multilingual:'多语言'})[value]||value;
function acceptBenchmarkMetadata(metadata){
 if(metadata&&Array.isArray(metadata.categories)&&Array.isArray(metadata.presets)){state.benchmarkMetadata=metadata;state.benchmarkMetadataWarning='';return}
 const categories=[...new Map(state.datasets.map(d=>[d.category_id||'unknown',{id:d.category_id||'unknown',label:d.category||'未分类',description:''}])).values()];
 state.benchmarkMetadata={categories,presets:[]};state.benchmarkMetadataWarning='分类说明暂时无法加载，已按评测集信息显示分类；可以继续筛选和勾选，请稍后点击刷新重试。';
}

function benchmarkTags(d){
 const tags=[...(d.scope?[d.scope==='demo'?'Demo · 小样本':'完整配置']:[]),...(d.languages||[]).map(languageName),...(d.tags||[])];
 if(d.requires_judge)tags.push('需要裁判');
 if(d.requires_execution)tags.push('执行代码');
 return `<div class="benchmark-tags">${[...new Set(tags)].map(t=>`<span>${esc(t)}</span>`).join('')}</div>`;
}
function datasetMatches(d){
 const p=datasetPicker;
 if(p.category!=='all'&&d.category_id!==p.category)return false;
 if(p.scope!=='all'&&d.scope!==p.scope)return false;
 if(p.language!=='all'&&!(d.languages||[]).includes(p.language))return false;
 if(p.availability==='ready'&&!d.available||p.availability==='blocked'&&d.available)return false;
 const query=p.query.trim().toLowerCase();
 return !query||[d.id,d.label,d.category,d.purpose,...(d.suitable_uses||[]),...(d.tags||[])].join(' ').toLowerCase().includes(query);
}
function pickerSnapshot(){
 const filtered=state.datasets.filter(datasetMatches),pages=Math.max(1,Math.ceil(filtered.length/datasetPicker.size));
 datasetPicker.page=Math.min(Math.max(1,datasetPicker.page),pages);
 const start=(datasetPicker.page-1)*datasetPicker.size;
 return {filtered,pages,start,visible:filtered.slice(start,start+datasetPicker.size)};
}
function categoryFilters({scores=false}={}){
 const rows=scores?state.detail?.scores||[]:state.datasets;
 const current=scores?resultPaging.category:datasetPicker.category;
 const attr=scores?'data-score-category':'data-picker-category';
 const count=id=>rows.filter(row=>((scores?scoreInterpretation(row):row).category_id||'unknown')===id).length;
 const categories=[{id:'all',label:'全部',count:rows.length},...benchmarkCategories().map(c=>({...c,count:count(c.id)}))];
 if(scores&&rows.some(s=>!benchmarkCategories().some(c=>c.id===scoreInterpretation(s).category_id)))categories.push({id:'unknown',label:'未分类',count:count('unknown')});
 return categories.map(c=>`<button type="button" class="category-filter ${current===c.id?'active':''}" ${attr}="${esc(c.id)}" aria-pressed="${current===c.id}"><span>${esc(c.label)}</span><span class="category-count">${c.count}</span></button>`).join('');
}
function presetItems(preset){return preset.datasets.map(id=>state.datasets.find(d=>d.id===id)).filter(Boolean)}
function presetMarkup(){if(!benchmarkPresets().length)return state.benchmarkMetadataWarning?`<section class="panel full"><p class="metadata-warning">${esc(state.benchmarkMetadataWarning)}</p></section>`:'';return `<section class="panel full preset-panel"><div class="panel-head"><div><h2>按用途快速选择</h2><p class="hint">预设会替换当前选择；只选可运行项目。生成额度由下方参数决定。</p></div></div><div class="benchmark-presets">${benchmarkPresets().map(p=>{
 const items=presetItems(p),ready=items.filter(d=>d.available),blocked=items.filter(d=>!d.available);
 return `<article class="benchmark-preset ${datasetPicker.activePreset===p.id?'active':''}" data-preset-card="${esc(p.id)}"><h3>${esc(p.label)}</h3><p>${esc(p.description)}</p><span class="preset-count">可运行 ${ready.length} / ${items.length} 项</span><button type="button" class="secondary" data-benchmark-preset="${esc(p.id)}" ${!ready.length||benchmarkBusy()?'disabled':''}>${ready.length?'替换当前选择':'准备后可选'}</button><details><summary>查看组合与运行条件</summary><ul>${items.map(d=>`<li><b>${esc(d.label)}</b><span>${d.available?'可运行':esc(d.unsupported_reason||'尚未就绪')}</span></li>`).join('')}</ul>${blocked.length?`<p>应用时跳过 ${blocked.length} 个暂不可运行项目。</p>`:''}</details></article>`;
 }).join('')}</div></section>`}
function selectedMarkup(snapshot){
 const selected=state.selectedDatasets.map(id=>state.datasets.find(d=>d.id===id)).filter(Boolean);
 const filteredIds=new Set(snapshot.filtered.map(d=>d.id));
 const hidden=selected.filter(d=>!filteredIds.has(d.id)).length;
 const busy=benchmarkBusy();
 return `<div class="selection-summary"><div class="selection-heading"><strong id="selectedDatasetCount" role="status" aria-live="polite">已选 ${selected.length} 个${hidden?` · ${hidden} 个在筛选范围外`:''}</strong><span class="hint">切换分类、搜索或翻页会保留选择</span></div>${selected.length?`<div class="selected-datasets">${selected.map(d=>`<button type="button" data-remove-dataset="${esc(d.id)}" aria-label="移除 ${esc(d.label)}" ${busy?'disabled':''}>${esc(d.label)}<span aria-hidden="true">×</span></button>`).join('')}</div>`:''}<div class="form-actions"><button class="primary" id="startEvaluation" ${!selected.length||busy?'disabled':''}>${busy?'评测运行中…':'开始 OpenCompass 评测'}</button><button class="secondary" id="selectVisibleDatasets" ${busy||!snapshot.visible.some(d=>d.available)?'disabled':''}>选中本页可运行项</button><button class="secondary" id="clearSelectedDatasets" ${busy||!selected.length?'disabled':''}>清空选择</button></div></div>`;
}
function pickerResultsMarkup(snapshot){
 const category=benchmarkCategories().find(c=>c.id===datasetPicker.category);
 return `<div class="picker-category-note"><strong>${esc(category?.label||'全部评测能力')}</strong><p>${esc(category?.description||'按能力选择评测，结合各项用途和运行条件决定测试组合。Demo 用于验证流程，正式比较优先选择完整配置。')}</p></div><div class="picker-count" role="status" aria-live="polite">筛选到 ${snapshot.filtered.length} 个配置 · 当前页 ${snapshot.visible.length} 个 · 可运行 ${snapshot.filtered.filter(d=>d.available).length} 个</div><div class="dataset-grid">${snapshot.visible.map(d=>renderDatasetCard(d,state.selectedDatasets.includes(d.id))).join('')||'<div class="picker-empty"><strong>没有匹配的评测集</strong><p>试试其他分类、语言或运行状态，已选项目仍然保留。</p><button class="secondary" id="resetDatasetFilters">重置筛选</button></div>'}</div><nav class="pagination" aria-label="评测集分页"><span class="hint">${snapshot.filtered.length?`${snapshot.start+1}–${Math.min(snapshot.start+datasetPicker.size,snapshot.filtered.length)} / ${snapshot.filtered.length} 个配置`:'0 个配置'}</span><div class="page-controls"><button class="secondary" id="datasetPrev" ${datasetPicker.page===1?'disabled':''}>上一页</button><span class="hint">${datasetPicker.page} / ${snapshot.pages}</span><button class="secondary" id="datasetNext" ${datasetPicker.page===snapshot.pages?'disabled':''}>下一页</button></div></nav>`;
}
function rememberPickerDetails(){document.querySelectorAll('.dataset-grid .benchmark-info').forEach(el=>{if(el.open)datasetPicker.openInfo.add(el.dataset.benchmarkId);else datasetPicker.openInfo.delete(el.dataset.benchmarkId)})}
function resetPickerFilters(){Object.assign(datasetPicker,{category:'all',scope:'all',language:'all',availability:'all',query:'',page:1});renderStandardEvaluation()}
function updatePickerResults(){
 rememberPickerDetails();
 const snapshot=pickerSnapshot();
 $('#datasetCategoryFilters').innerHTML=categoryFilters();
 $('#datasetSelection').innerHTML=selectedMarkup(snapshot);
 $('#datasetResults').innerHTML=pickerResultsMarkup(snapshot);
 bindPickerResults(snapshot);
 if(typeof bindEnvironmentLinks==='function')bindEnvironmentLinks();
 document.querySelectorAll('[data-benchmark-preset]').forEach(btn=>{const preset=benchmarkPresets().find(p=>p.id===btn.dataset.benchmarkPreset);btn.disabled=benchmarkBusy()||!presetItems(preset||{datasets:[]}).some(d=>d.available)});
}
function setPickerSelection(ids){state.selectedDatasets=[...new Set(ids)].filter(id=>state.datasets.some(d=>d.id===id&&d.available));datasetPicker.activePreset=null;if($('#presetNotice'))$('#presetNotice').innerHTML='';updatePickerResults();document.querySelectorAll('[data-benchmark-preset]').forEach(btn=>btn.closest('.benchmark-preset').classList.remove('active'))}
function bindPickerResults(snapshot){
 document.querySelectorAll('[data-picker-category]').forEach(btn=>btn.onclick=()=>{datasetPicker.category=btn.dataset.pickerCategory;datasetPicker.page=1;updatePickerResults();[...document.querySelectorAll('[data-picker-category]')].find(el=>el.dataset.pickerCategory===datasetPicker.category)?.focus({preventScroll:true})});
 document.querySelectorAll('[data-dataset]').forEach(input=>{
   input.disabled=!state.datasets.find(d=>d.id===input.dataset.dataset)?.available||benchmarkBusy();
   input.onchange=()=>{const id=input.dataset.dataset;const selected=new Set(state.selectedDatasets);input.checked?selected.add(id):selected.delete(id);setPickerSelection([...selected]);[...document.querySelectorAll('[data-dataset]')].find(el=>el.dataset.dataset===id)?.focus({preventScroll:true})};
 });
 document.querySelectorAll('.dataset-grid .benchmark-info').forEach(el=>{el.open=datasetPicker.openInfo.has(el.dataset.benchmarkId);el.ontoggle=()=>{if(el.open)datasetPicker.openInfo.add(el.dataset.benchmarkId);else datasetPicker.openInfo.delete(el.dataset.benchmarkId)}});
 document.querySelectorAll('[data-remove-dataset]').forEach(btn=>btn.onclick=()=>setPickerSelection(state.selectedDatasets.filter(id=>id!==btn.dataset.removeDataset)));
 $('#startEvaluation').onclick=startEvaluation;
 $('#clearSelectedDatasets').onclick=()=>setPickerSelection([]);
 $('#selectVisibleDatasets').onclick=()=>setPickerSelection([...state.selectedDatasets,...snapshot.visible.filter(d=>d.available).map(d=>d.id)]);
 for(const [id,delta]of[['datasetPrev',-1],['datasetNext',1]])$('#'+id).onclick=()=>{datasetPicker.page+=delta;updatePickerResults();$('#datasetResults').scrollIntoView({block:'start'});const button=$('#'+id);(button.disabled?$('#'+(delta>0?'datasetPrev':'datasetNext')):button).focus({preventScroll:true})};
 if($('#resetDatasetFilters'))$('#resetDatasetFilters').onclick=resetPickerFilters;
 const save=$('#saveCollection');if(save)save.disabled=!state.selectedDatasets.length;
}
function applyBenchmarkPreset(id){
 const preset=benchmarkPresets().find(p=>p.id===id);if(!preset||benchmarkBusy())return;
 const items=presetItems(preset),ready=items.filter(d=>d.available),blocked=items.filter(d=>!d.available);
 if(!ready.length){showToast('组合尚未准备好，请查看各项运行条件');return}
 state.selectedDatasets=ready.map(d=>d.id);datasetPicker.activePreset=id;
 renderStandardEvaluation();
 $('#presetNotice').innerHTML=`<strong>已选择「${esc(preset.label)}」中的 ${ready.length} 项。</strong>${blocked.length?`<details open><summary>另有 ${blocked.length} 项暂不可运行，已跳过</summary><ul>${blocked.map(d=>`<li>${esc(d.label)}：${esc(d.unsupported_reason||'尚未准备好')}</li>`).join('')}</ul></details>`:''}`;
}
function renderStandardEvaluation(){
 rememberPickerDetails();
 const running=state.evalStatus?.status==='running';
 const collections=state.collections.map(c=>`<div class="collection-card"><div><h3>${esc(c.name)}</h3><p>${esc(c.description||'')} · ${c.datasets.length} 个评测集<br/>${esc(collectionNames(c))}</p></div><button class="secondary" data-apply-collection="${esc(c.id)}" ${running?'disabled':''}>替换当前选择</button></div>`).join('');
 $('#crumb').textContent='开始评测';$('#actions').innerHTML='<button class="refresh" id="refreshEvaluate">刷新</button>';
 $('#app').innerHTML=evaluationModes()+`<section class="hero eval-intro"><div><h1>按能力，选择一次评测。</h1><p class="muted">当前模型：${esc(state.settings?.model||'尚未配置')} · 先选用途组合，或按分类逐项选择。</p></div><button class="secondary" id="editModel">模型设置</button></section><div id="evaluationStatus" aria-live="polite">${evaluationStatusMarkup()}</div><div class="content">${presetMarkup()}<section class="panel full dataset-explorer"><div class="panel-head"><div><h2>评测集目录</h2><p class="hint">共 ${state.datasets.length} 个配置，${state.datasets.filter(d=>d.available).length} 个可运行。分类数字为该能力下的全部配置数。</p></div></div><div class="category-filters" id="datasetCategoryFilters" role="group" aria-label="按评测能力筛选"></div><div class="dataset-toolbar"><div class="field dataset-search"><label for="datasetSearch">搜索评测集或用途</label><input id="datasetSearch" type="search" placeholder="例如：GPQA、中文、长报告" value="${esc(datasetPicker.query)}"/></div><div class="field"><label for="datasetScope">评测规模</label><select id="datasetScope"><option value="all">全部规模</option><option value="full">完整配置</option><option value="demo">Demo 小样本</option></select></div><div class="field"><label for="datasetLanguage">语言</label><select id="datasetLanguage"><option value="all">全部语言</option><option value="zh">中文</option><option value="en">英文</option></select></div><div class="field"><label for="datasetAvailability">运行条件</label><select id="datasetAvailability"><option value="all">全部状态</option><option value="ready">可运行</option><option value="blocked">待准备</option></select></div><div class="field"><label for="datasetPageSize">每页</label><select id="datasetPageSize"><option value="8">8 个</option><option value="16">16 个</option><option value="32">32 个</option></select></div></div><div id="datasetSelection"></div><div id="presetNotice" class="preset-notice" role="status"></div><details id="abilityParameters" class="ability-parameters" ${datasetPicker.optionsOpen?'open':''}><summary>生成参数 · 上下文 ${state.abilityOptions.max_seq_len.toLocaleString()} / 生成额度 ${state.abilityOptions.max_out_len.toLocaleString()} tokens</summary><div class="form-grid"><div class="field"><label for="abilityContext">上下文窗口上限 tokens</label><input id="abilityContext" type="number" min="128" max="262144" value="${state.abilityOptions.max_seq_len}" ${running?'disabled':''}/></div><div class="field"><label for="abilityOutput">每题生成额度 tokens（含思考）</label><input id="abilityOutput" type="number" min="64" max="65536" value="${state.abilityOptions.max_out_len}" ${running?'disabled':''}/></div></div><p class="hint">输入和生成内容共用上下文窗口；思考耗尽额度或截断会影响分数。参数随结果保存，服务端也可能限制实际长度。</p></details><div id="datasetResults"></div></section><section class="panel full"><div class="panel-head"><h2>我的评测集合</h2><span class="hint">保存当前选择，便于下次复用</span></div><div class="collection-list">${collections||'<div class="empty">暂无自定义集合。选择评测集后可以保存常用组合。</div>'}</div><div class="form-grid" style="margin-top:18px"><div class="field"><label for="collectionName">集合名称</label><input id="collectionName" value="${esc(state.collectionDraft.name)}" placeholder="例如：模型升级回归"/></div><div class="field"><label for="collectionDescription">说明</label><input id="collectionDescription" value="${esc(state.collectionDraft.description)}" placeholder="可选"/></div></div><div class="form-actions"><button class="secondary" id="saveCollection" ${!state.selectedDatasets.length?'disabled':''}>保存当前选择</button></div></section></div>`;
 bindModes();bindStatus();
 $('#editModel').onclick=()=>navigate('settings');$('#refreshEvaluate').onclick=()=>loadEvaluate();
 $('#datasetSearch').oninput=e=>{datasetPicker.query=e.target.value;datasetPicker.page=1;updatePickerResults()};
 for(const [id,key]of[['datasetScope','scope'],['datasetLanguage','language'],['datasetAvailability','availability'],['datasetPageSize','size']]){const el=$('#'+id);el.value=String(datasetPicker[key]);el.onchange=()=>{datasetPicker[key]=key==='size'?Number(el.value):el.value;datasetPicker.page=1;updatePickerResults()}}
 $('#abilityParameters').ontoggle=e=>datasetPicker.optionsOpen=e.target.open;
 $('#abilityContext').oninput=e=>state.abilityOptions.max_seq_len=Number(e.target.value);
 $('#abilityOutput').oninput=e=>state.abilityOptions.max_out_len=Number(e.target.value);
 $('#collectionName').oninput=e=>state.collectionDraft.name=e.target.value;$('#collectionDescription').oninput=e=>state.collectionDraft.description=e.target.value;$('#saveCollection').onclick=saveCollection;
 document.querySelectorAll('[data-benchmark-preset]').forEach(btn=>btn.onclick=()=>applyBenchmarkPreset(btn.dataset.benchmarkPreset));
 document.querySelectorAll('[data-apply-collection]').forEach(btn=>btn.onclick=()=>{const collection=state.collections.find(c=>c.id===btn.dataset.applyCollection);if(collection){state.selectedDatasets=collection.datasets.filter(id=>state.datasets.some(d=>d.id===id&&d.available));datasetPicker.activePreset=null;renderStandardEvaluation();showToast('已替换为集合中可运行的项目')}});
 updatePickerResults();
}

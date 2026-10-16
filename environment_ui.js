const environmentView={data:null,loading:false,starting:false,error:'',timer:null,logOpen:false};
const environmentWorking=()=>environmentView.starting||environmentView.data?.busy||environmentView.data?.status==='running';
const environmentStatusLabel=value=>({idle:'尚未安装',running:'处理中',completed:'已完成',failed:'失败',interrupted:'已中断',ready:'已就绪',partial:'部分完成',pending:'等待中',skipped:'已跳过'})[value]||value||'待检查';

function environmentChecks(data){
 const runtime=data.runtime||{},code=data.code||{};
 const checks=[
   {label:'容器服务',ready:runtime.running,text:runtime.message||(runtime.running?'已启动':runtime.installed?'已安装，等待启动':'尚未安装容器服务')},
   {label:'Python 代码评测镜像',ready:Boolean(code.image_id),text:code.image_id?'已准备':`安装时准备 ${code.image||'代码评测镜像'}`},
   {label:'隔离执行自检',ready:code.self_test_passed,text:code.self_test_passed?'通过正确、错误与超时测试':'通过自检后才会启用代码评测'},
   ...(code.data||[]).map(d=>({label:d.label||d.id,ready:d.ready,text:d.message||(d.ready?'数据已准备':'等待准备测试数据')}))
 ];
 return `<div class="environment-checks">${checks.map(c=>`<article class="environment-check"><div><h3>${esc(c.label)}</h3><span class="environment-badge ${c.ready?'ready':'pending'}">${c.ready?'已就绪':'待准备'}</span></div><p>${esc(c.text)}</p></article>`).join('')}</div>`;
}
function environmentProgress(data){
 const steps=data.steps||[],logs=data.logs||[];
 if(!data.action&&!steps.length&&!logs.length)return '';
 return `<section class="panel full environment-progress"><div class="panel-head"><h2>安装与检测进度</h2><span class="environment-badge ${data.status==='failed'?'failed':data.status==='running'||data.warnings?.length?'pending':'ready'}">${esc(environmentStatusLabel(data.status==='completed'&&data.warnings?.length?'partial':data.status))}</span></div>${data.step?`<p class="environment-current" role="status">${esc(steps.find(step=>step.id===data.step)?.label||data.step)}</p>`:''}${data.error?`<p class="environment-error" role="alert">${esc(data.error)}</p>`:''}${(data.warnings||[]).map(message=>`<p class="environment-notice">${esc(message)}</p>`).join('')}<ol class="environment-steps">${steps.map((step,i)=>`<li class="${esc(step.status||'pending')}"><span class="environment-step-number" aria-hidden="true">${i+1}</span><div><strong>${esc(step.label||step.id)}</strong><span>${esc(environmentStatusLabel(step.status))}</span>${step.message?`<p>${esc(step.message)}</p>`:''}</div></li>`).join('')}</ol><details id="environmentLogs" ${environmentView.logOpen||data.status==='failed'?'open':''}><summary>查看安装日志（最近 ${logs.length} 条）</summary><pre>${esc(logs.map(row=>typeof row==='string'?row:(row.at?row.at+' ':'')+(row.message||'')).join('\n')||'暂时没有日志。')}</pre></details></section>`;
}
function renderEnvironmentView(){
 const main=document.querySelector('main'),previousScroll=main?.scrollTop;
 const data=environmentView.data,busy=environmentWorking(),runtime=data?.runtime||{},code=data?.code||{},allDataReady=Boolean(code.data?.length)&&code.data.every(item=>item.ready);
 $('#crumb').textContent='环境管理';$('#actions').innerHTML='<button class="refresh" id="refreshEnvironment">刷新状态</button>';
 $('#app').innerHTML=`<section class="hero eval-intro"><div><h1>准备代码评测环境。</h1><p class="muted">安装在运行本网页后台的电脑；模型继续通过已保存的局域网接口请求。</p></div><button class="secondary" id="environmentToBenchmarks">查看编程评测</button></section>${environmentView.error?`<section class="panel full"><p class="environment-error" role="alert">${esc(environmentView.error)}</p></section>`:''}${!data?`<section class="panel full"><p role="status">${environmentView.loading?'正在检查容器服务和代码评测条件…':'尚未获取环境状态，请点击刷新状态。'}</p></section>`:`<div class="content"><section class="panel full"><div class="panel-head"><div><h2>基础 Python 代码评测</h2><p class="hint">准备隔离执行环境和 MBPP 数据；HumanEval CN 还需本地中文数据文件。</p></div><span class="environment-badge ${code.ready?'ready':'pending'}">${code.ready?'执行环境已就绪':'需要准备'}</span></div>${environmentChecks(data)}${!code.ready&&code.message?`<p class="environment-notice">${esc(code.message)}</p>`:''}<div class="form-actions">${!runtime.running?`<button class="secondary" id="startCodeRuntime" ${busy||!data.can_start_runtime?'disabled':''}>${runtime.installed?'启动容器服务':'容器服务尚未安装'}</button>`:''}<button class="primary" id="installCodeEnvironment" ${busy||(code.ready&&allDataReady)||!data.can_install?'disabled':''}>${busy?'正在处理…':code.ready&&allDataReady?'代码评测环境已就绪':data.status==='failed'||code.ready?'安装 / 修复代码评测环境':'安装代码评测环境'}</button><button class="secondary" id="checkCodeEnvironment" ${busy?'disabled':''}>重新检测</button></div><p class="hint">首次安装需要联网下载 Python 镜像和公开测试数据，安装进度会持续保存；可以切换页面后再回来查看。</p>${!runtime.installed?'<p class="environment-error">请先在这台电脑上安装并启动 OrbStack 或 Docker Desktop，再返回检测。本页面可以启动已安装的容器服务。</p>':''}</section>${environmentProgress(data)}<section class="panel full environment-coverage"><div class="panel-head"><h2>安装后能运行哪些评测</h2></div><div class="environment-supported"><div><strong>HumanEval CN</strong><p>根据中文函数说明生成代码，在容器内执行功能测试。需提供真实 HumanEval CN 数据；具体缺失条件见上方检查。</p></div><div><strong>MBPP</strong><p>根据编程要求生成 Python 小程序，在容器内运行断言测试。官方数据可在安装时自动下载。</p></div></div><p class="hint">HumanEval+、MBPP+、LiveCodeBench 和 BFCL 需要各自的评分或工具执行适配，当前仍会显示待准备。裁判模型、长上下文 tokenizer 的准备条件与本环境分别检查。</p></section></div>`}`;
 $('#refreshEnvironment').onclick=()=>loadEnvironment(true);
 $('#environmentToBenchmarks').onclick=()=>{Object.assign(datasetPicker,{category:'code',scope:'all',language:'all',availability:'all',query:'',page:1});navigate('evaluate')};
 if($('#startCodeRuntime'))$('#startCodeRuntime').onclick=()=>startEnvironmentAction('start_runtime');
 if($('#installCodeEnvironment'))$('#installCodeEnvironment').onclick=()=>startEnvironmentAction('install_code');
 if($('#checkCodeEnvironment'))$('#checkCodeEnvironment').onclick=()=>startEnvironmentAction('check');
 if($('#environmentLogs'))$('#environmentLogs').ontoggle=e=>environmentView.logOpen=e.target.open;
 if(main&&previousScroll!=null)main.scrollTop=previousScroll;
}
async function loadEnvironment(force=false){
 if(environmentView.loading&&!force)return;
 environmentView.loading=true;
 if(state.view==='environment'&&!environmentView.data)renderEnvironmentView();
 try{environmentView.data=await request('/api/environment/status');environmentView.error='';if(state.view==='environment')renderEnvironmentView();pollEnvironment()}
 catch(error){environmentView.error='环境状态获取失败：'+error.message;if(state.view==='environment')renderEnvironmentView()}
 finally{environmentView.loading=false;if(state.view==='environment')renderEnvironmentView()}
}
async function startEnvironmentAction(action){
 if(environmentWorking())return;
 environmentView.starting=true;environmentView.error='';renderEnvironmentView();
 try{environmentView.data=await request('/api/environment/install',{method:'POST',body:JSON.stringify({action})});showToast('环境任务已开始');pollEnvironment()}
 catch(error){environmentView.error=error.message}
 finally{environmentView.starting=false;if(state.view==='environment')renderEnvironmentView()}
}
function pollEnvironment(){
 if(environmentView.timer||!(environmentView.data?.status==='running'||environmentView.data?.busy))return;
 environmentView.timer=setTimeout(async()=>{
   try{
     environmentView.data=await request('/api/environment/status');environmentView.error='';
     if(environmentView.data.status!=='running'&&!environmentView.data.busy){
       state.datasets=await request('/api/datasets');state.selectedDatasets=state.selectedDatasets.filter(id=>state.datasets.some(d=>d.id===id&&d.available));
       showToast(environmentView.data.code?.ready?'代码评测环境已就绪':environmentView.data.status==='failed'?'环境任务失败，可查看原因并重试':'环境检测已更新');
       if(state.view==='evaluate'&&state.evalMode==='standard')renderStandardEvaluation();
     }
     if(state.view==='environment')renderEnvironmentView();
   }catch(error){environmentView.error='进度更新失败：'+error.message;if(state.view==='environment')renderEnvironmentView()}
   finally{environmentView.timer=null;pollEnvironment()}
 },2000);
}
function bindEnvironmentLinks(){document.querySelectorAll('[data-open-environment]').forEach(button=>button.onclick=()=>navigate('environment'))}

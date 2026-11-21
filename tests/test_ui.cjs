const {test}=require('node:test');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const path=require('node:path');
const fs=require('node:fs');
const net=require('node:net');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'..');
async function until(fn,label){const end=Date.now()+15000;while(Date.now()<end){if(await fn())return;await new Promise(r=>setTimeout(r,30));}throw Error('Timed out: '+label);}
async function freePort(){const s=net.createServer();await new Promise(r=>s.listen(0,'127.0.0.1',r));const port=s.address().port;await new Promise(r=>s.close(r));return port;}
test('local app diagnostics, filtering, example inference and threshold isolation',{timeout:60000},async()=>{
 const python=process.env.EXOFINDER_PYTHON||[path.join(root,'.venv/bin/python'),path.resolve(root,'../../work/.venv/bin/python')].find(p=>fs.existsSync(p));
 assert.ok(python,'Create the Python environment or set EXOFINDER_PYTHON.');
 const port=await freePort(),base=`http://127.0.0.1:${port}`;
 const server=spawn(python,['-m','uvicorn','exofinder.server:app','--host','127.0.0.1','--port',String(port)],{cwd:root,stdio:'ignore'});
 let dom;const errors=[];
 try{
  await until(async()=>{try{return(await fetch(base+'/api/health')).ok;}catch{return false;}},'server readiness');
  const vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e.message));
  dom=await JSDOM.fromURL(base,{resources:'usable',runScripts:'dangerously',virtualConsole:vc,beforeParse(w){w.fetch=(url,opts)=>fetch(new URL(url,base),opts);w.HTMLElement.prototype.scrollIntoView=function(){};}});
  const w=dom.window,d=w.document,$=s=>d.querySelector(s);
  const change=(id,value)=>{const e=$(id);e.value=value;e.dispatchEvent(new w.Event('change',{bubbles:true}));};
  await until(()=>$('#target-title').textContent!=='Light curve'&&$('#error-table tbody tr'),'loaded UI');
  assert.equal($('#error-table tbody').children.length,14);
  assert.ok($('#pr-chart svg'));assert.ok($('#calibration-chart svg'));
  change('#outcome','missed_candidate');
  await until(()=>$('#targets').children.length===7,'missed-candidate filter');
  change('#outcome','');change('#sort','high');
  const scores=[...d.querySelectorAll('.target-score')].map(e=>Number(e.textContent));assert.deepEqual(scores,[...scores].sort((a,b)=>b-a));
  $('#search').value='NO_SUCH_OBJECT';$('#search').dispatchEvent(new w.Event('input'));
  assert.equal($('.signal').hidden,true);assert.match($('#targets').textContent,/No targets/);
  $('.nav[data-page="evaluation"]').click();assert.ok($('#evaluation').classList.contains('active'));
  const frozen=$('#comparison').textContent;
  $('#threshold-slider').value='1';$('#threshold-slider').dispatchEvent(new w.Event('input'));
  await until(()=>$('#threshold-result').textContent.includes('0/71'),'validation threshold result');
  assert.equal($('#comparison').textContent,frozen);
  $('#threshold-reset').click();await until(()=>!$('#threshold-result').textContent.includes('Calculating'),'threshold reset');
  $('.inspect-error').click();await until(()=>$('#target-title').textContent!=='Light curve'&&$('.signal').hidden===false,'error selection');
  assert.equal($('#split').value,'test');assert.ok($('#explorer').classList.contains('active'));
  $('#load-example').click();await until(()=>$('#upload-result').textContent.includes('loaded'),'example load');
  $('#upload-form').dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
  await until(()=>$('#download-prediction'),'inference completion');
  assert.match($('#upload-result').textContent,/0\.648/);assert.ok($('#upload-plot svg'));
  assert.equal($('#predict-button').disabled,false);
  assert.deepEqual(errors,[]);
 }finally{if(dom)dom.window.close();server.kill('SIGTERM');await new Promise(resolve=>{if(server.exitCode!==null)return resolve();server.once('exit',resolve);});}
});

'use strict';
const $ = s => document.querySelector(s);
const escapeHTML = v => String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = n => (n*100).toFixed(1)+'%';
const number = (n,d=2) => Number(n).toFixed(d);
let overview, current, view='local', requestCounter=0, exampleCSV=null, latestPrediction=null;
async function api(url,options){const r=await fetch(url,{cache:'no-store',...options});const j=await r.json();if(!r.ok)throw Error(typeof j.detail==='string'?j.detail:JSON.stringify(j.detail));return j;}
function plotSVG(values,mask,low,high,label,width){
 const w=Math.max(300,width||$('#plot').clientWidth),h=245,p={l:52,r:18,t:16,b:43};
 const valid=values.filter((_,i)=>mask[i]);
 if(!valid.length)return '<p>No populated bins.</p>';
 let ymin=Math.min(...valid),ymax=Math.max(...valid);const pad=Math.max((ymax-ymin)*.16,.3);ymin-=pad;ymax+=pad;
 const x=i=>p.l+(i+.5)/values.length*(w-p.l-p.r), y=v=>p.t+(ymax-v)/(ymax-ymin)*(h-p.t-p.b);
 let grid='',dots='';
 for(let i=0;i<5;i++){const v=ymin+(ymax-ymin)*i/4;grid+=`<line x1="${p.l}" y1="${y(v)}" x2="${w-p.r}" y2="${y(v)}" stroke="#26333d"/><text x="${p.l-10}" y="${y(v)+4}" text-anchor="end" fill="#9daebc" font-size="11">${v.toFixed(1)}</text>`;}
 for(let i=0;i<5;i++){const xx=p.l+i/4*(w-p.l-p.r),v=low+(high-low)*i/4;grid+=`<text x="${xx}" y="${h-23}" text-anchor="middle" fill="#9daebc" font-size="11">${v.toFixed(2)}</text>`;}
 const center=p.l+(0-low)/(high-low)*(w-p.l-p.r);grid+=`<line x1="${center}" y1="${p.t}" x2="${center}" y2="${h-p.b}" stroke="#68776e" stroke-dasharray="3 5"/>`;
 values.forEach((v,i)=>{if(mask[i])dots+=`<circle aria-hidden="true" cx="${x(i)}" cy="${y(v)}" r="${values.length>200?1.8:2.8}" fill="#8dd6ca" opacity=".85"><title>Phase ${(low+(i+.5)/values.length*(high-low)).toFixed(4)}; flux ${v.toFixed(3)} σ</title></circle>`;});
 return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(label)}" style="font-family:ui-monospace,monospace">${grid}${dots}<text x="${w/2}" y="${h-2}" fill="#9daebc" font-size="11" text-anchor="middle">${label.includes('hours')?'Time from transit center (hours)':'Orbital phase (cycles)'}</text></svg>`;
}
function renderPlot(){if(!current)return;const o=current.object;const data=current[view+'_view'];const extent=view==='local'?o.local_extent_days*24:.5;$('#plot').innerHTML=plotSVG(data[0],data[1],-extent,extent,view==='local'?'Transit close-up in hours':'Full orbital phase');}
function renderTargets(){
 const q=$('#search').value.toLowerCase(),split=$('#split').value,label=$('#label').value;
 let rows=overview.objects.filter(o=>(!split||o.split===split)&&(label===''||String(o.label)===label)&&[o.kepoi_name,o.kepid,o.kepler_name].join(' ').toLowerCase().includes(q));
 const outcome=$('#outcome').value, threshold=overview.metrics.threshold;
 rows=rows.filter(o=>!outcome||(outcome==='mismatch'&&Number(o.score>=threshold)!==o.label)||(outcome==='false_alarm'&&o.label===0&&o.score>=threshold)||(outcome==='missed_candidate'&&o.label===1&&o.score<threshold));
 const sort=$('#sort').value;
 rows.sort((a,b)=>sort==='high'?b.score-a.score:sort==='low'?a.score-b.score:sort==='borderline'?Math.abs(a.score-threshold)-Math.abs(b.score-threshold):a.kepoi_name.localeCompare(b.kepoi_name));
 $('#count').textContent=rows.length;
 const scroll=$('#targets').scrollTop;
 $('.signal').hidden=!rows.length;
 if(!rows.length){requestCounter++;current=null;}

 $('#targets').innerHTML=rows.length?rows.map(o=>`<button class="target ${current?.object.kepoi_name===o.kepoi_name?'selected':''}" data-id="${escapeHTML(o.kepoi_name)}" aria-pressed="${current?.object.kepoi_name===o.kepoi_name}"><span><strong>${escapeHTML(o.kepoi_name)}</strong><small>${o.label?'Candidate':'False positive'} · ${number(o.koi_period)} d</small></span><span class="target-score ${o.score<overview.metrics.threshold?'low':''}">${number(o.score,3)}</span></button>`).join(''):'<p class="empty">No targets match these filters.</p>';
 $('#targets').scrollTop=scroll;
 document.querySelectorAll('.target').forEach(b=>b.onclick=()=>selectTarget(b.dataset.id));
 if(rows.length&&!rows.some(o=>o.kepoi_name===current?.object.kepoi_name))selectTarget(rows[0].kepoi_name);
}
async function selectTarget(id){
 const ticket=++requestCounter;
 try{const result=await api('/api/objects/'+encodeURIComponent(id));if(ticket!==requestCounter)return;current=result;
 const o=current.object;
 $('#target-kic').textContent=`KIC ${o.kepid} / QUARTER 9`;
 $('#target-title').textContent=o.kepler_name||o.kepoi_name;
 $('#target-label').textContent=o.label?'Catalog: candidate':'Catalog: false positive';$('#target-label').classList.toggle('fp',!o.label);
 $('#facts').innerHTML=[['Orbital period',number(o.koi_period,3),'days'],['Transit duration',number(o.koi_duration),'hours'],['Clean observations',Number(o.cadences).toLocaleString(),'cadences'],['Flux noise',number(o.noise_ppm,0),'ppm']].map(([k,v,u])=>`<div class="fact"><span>${k}</span><strong>${v} <small>${u}</small></strong></div>`).join('');
 $('#score').textContent=number(o.score,3);$('#verdict').textContent=o.score>=current.threshold?'Candidate-like':'False-positive-like';
 $('#scale-fill').style.width=(o.score*100)+'%';$('#threshold-tick').style.left=(current.threshold*100)+'%';$('#threshold-label').textContent=`Validation-selected threshold ${number(current.threshold,3)}`;
 $('#split-note').textContent=o.split==='test'?'Held-out test object · never used to fit or select this model':o.split==='validation'?'Validation object · used for model and threshold selection':'Training object · used to fit this model';
 $('#source-link').href=o.url;renderPlot();renderTargets();
 }catch(e){$('#status').textContent=e.message;$('#status').classList.add('error');}
}
function renderEvaluation(){const m=overview.metrics;
 $('#metric-cards').innerHTML=[['Test average precision',pct(m.cnn.average_precision),`95% bootstrap CI ${pct(m.test_ap_95ci[0])}–${pct(m.test_ap_95ci[1])}`],['Test ROC AUC',number(m.cnn.roc_auc,3),'Ranking across decision thresholds'],['Candidate recall',pct(m.cnn.recall),'At validation-selected threshold'],['Unseen test stars',m.counts.test,`${pct(m.test_prevalence)} catalog candidates`]].map(([a,b,c])=>`<div class="metric-card"><span>${a}</span><strong>${b}</strong><small>${c}</small></div>`).join('');
 $('#comparison').innerHTML='<table><thead><tr><th>Model</th><th>AP</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>'+[['Dual-view CNN',m.cnn],['Logistic baseline',m.baseline]].map(([name,v])=>`<tr><td>${name}</td>${['average_precision','precision','recall','f1'].map(k=>`<td>${number(v[k],3)}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
 const cm=m.cnn.confusion_matrix;
 $('#matrix').innerHTML=`<div class="matrix-wrap"><span>Actual / predicted</span><span>False positive</span><span>Candidate</span><span>False positive</span><div class="matrix-cell">${cm[0][0]}</div><div class="matrix-cell wrong">${cm[0][1]}</div><span>Candidate</span><div class="matrix-cell wrong">${cm[1][0]}</div><div class="matrix-cell">${cm[1][1]}</div></div>`;
 const history=overview.history,w=500,h=170;const points=history.map((v,i)=>`${35+i/Math.max(history.length-1,1)*450},${135-v.validation_ap*110}`).join(' ');
 $('#learning').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Validation average precision by training epoch"><line x1="35" y1="135" x2="485" y2="135" stroke="#3b4853"/><line x1="35" y1="25" x2="485" y2="25" stroke="#26333d"/><text x="5" y="30" fill="#9daebc" font-size="11">1.0</text><text x="5" y="138" fill="#9daebc" font-size="11">0.0</text><polyline points="${points}" fill="none" stroke="#8dd6ca" stroke-width="2"/><text x="35" y="160" fill="#9daebc" font-size="11">Epoch 1</text><text x="485" y="160" fill="#9daebc" text-anchor="end" font-size="11">Epoch ${history.length}</text></svg>`;
 $('#run-info').innerHTML=Object.entries({'Run':m.run,'Training device':m.device.toUpperCase(),'Train / validation / test':Object.values(m.counts).join(' / '),'Parameters':m.parameters.toLocaleString(),'Best epoch':m.best_epoch,'Seed':m.seed,'Training duration':number(m.seconds,1)+' seconds','Dataset SHA-256':m.dataset_sha256.slice(0,20)+'…'}).map(([k,v])=>`<dt>${k}</dt><dd>${escapeHTML(v)}</dd>`).join('');
 $('#limitations').innerHTML=m.limitations.map(s=>`<li>${escapeHTML(s)}</li>`).join('');
}
document.querySelectorAll('.nav').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav').forEach(n=>{n.classList.toggle('active',n===b);if(n===b)n.setAttribute('aria-current','page');else n.removeAttribute('aria-current');});document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===b.dataset.page));});
window.addEventListener('resize',()=>renderPlot());
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(n=>{n.classList.toggle('selected',n===b);n.setAttribute('aria-pressed',n===b);});renderPlot();});
$('#search').addEventListener('input',()=>overview&&renderTargets());['split','label','outcome','sort'].forEach(id=>$('#'+id).addEventListener('change',()=>overview&&renderTargets()));
$('.intro-actions a').addEventListener('click',()=>$('.nav[data-page="explorer"]').click());
$('#open-upload').onclick=()=>$('#upload-dialog').showModal();$('#close-upload').onclick=()=>$('#upload-dialog').close();
$('#upload-form').onsubmit=async e=>{e.preventDefault();const button=$('#predict-button'),out=$('#upload-result');out.classList.remove('error');button.disabled=true;out.textContent='Preparing the light curve and running inference…';try{
 const file=$('#curve-file').files[0];if(!file&&!exampleCSV)throw Error('Choose a CSV file or load the real example.');if(file?.size>8_000_000)throw Error('Use a CSV smaller than 8 MB.');
 const {time,flux}=ExoCSV.parse(file?await file.text():exampleCSV);
 const result=await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({time,flux,period:Number($('#period').value),epoch:Number($('#epoch').value),duration:Number($('#duration').value)})});
 latestPrediction=result;
 out.innerHTML=`<div class="inference-scores"><div><span>CNN candidate score</span><strong>${number(result.score,3)}</strong></div><div><span>Baseline score</span><strong>${number(result.baseline_score,3)}</strong></div></div><p>${result.score>=result.threshold?'Candidate-like':'False-positive-like'} at saved threshold ${number(result.threshold,3)}.</p><div id="upload-plot" class="plot"></div><p>${result.info.cadences.toLocaleString()} retained observations · ${result.info.discarded_observations} removed/duplicates · ${pct(result.info.local_coverage)} local-bin coverage</p>${result.in_training_period_range?'':'<p class="domain-warning">Outside the training period range. This prediction is unvalidated for this domain.</p>'}<p>${escapeHTML(result.warning)}</p><button class="button secondary" id="download-prediction" type="button">Download inference JSON</button>`;
 $('#upload-plot').innerHTML=plotSVG(result.local_view[0],result.local_view[1],-result.info.local_extent_days*24,result.info.local_extent_days*24,'Uploaded transit close-up in hours',out.clientWidth);
 $('#download-prediction').onclick=()=>downloadJSON(latestPrediction,'exoplanet-inference.json');
 }catch(e){out.textContent=e.message;out.classList.add('error');}finally{button.disabled=false;}};
(async()=>{try{overview=await api('/api/overview');const m=overview.metrics;$('#status').textContent=`REAL KEPLER DATA / ${overview.objects.length} stars / ${m.counts.test} held-out test stars / trained on ${m.device.toUpperCase()} / pilot experiment`;
 $('#dataset-note').textContent=`DR25 · Q9 · ${overview.objects.length} usable targets`;renderEvaluation();renderTargets();loadDiagnostics();loadResearch();}catch(e){$('#status').textContent=e.message;$('#status').classList.add('error');}})();

function downloadJSON(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('#curve-file').addEventListener('change',()=>{exampleCSV=null;$('#upload-result').textContent='';});
$('#load-example').onclick=async()=>{const button=$('#load-example');button.disabled=true;try{const sample=await api('/api/example');exampleCSV=sample.csv;$('#curve-file').value='';['period','epoch','duration'].forEach(k=>$('#'+k).value=sample[k]);$('#upload-result').classList.remove('error');$('#upload-result').textContent=sample.name+' loaded. Run inference to inspect its score.';}catch(e){$('#upload-result').textContent=e.message;$('#upload-result').classList.add('error');}finally{button.disabled=false;}};

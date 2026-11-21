'use strict';
let diagnosticsData, thresholdRequest=0, thresholdTimer;
const metricText = n => n===null?'—':pct(n);
function unitChart(container,series,xLabel,yLabel,{horizontal=null,diagonal=false,intervals=[]}={}){
 const w=Math.max(300,container.clientWidth||540),h=265,p={l:47,r:16,t:18,b:47};
 const x=v=>p.l+v*(w-p.l-p.r),y=v=>h-p.b-v*(h-p.t-p.b);
 let svg='';
 for(let i=0;i<=4;i++){const v=i/4;svg+=`<line x1="${x(0)}" y1="${y(v)}" x2="${x(1)}" y2="${y(v)}" stroke="#28343e"/><text x="${x(0)-8}" y="${y(v)+4}" text-anchor="end">${v.toFixed(2)}</text><text x="${x(v)}" y="${h-25}" text-anchor="middle">${v.toFixed(2)}</text>`;}
 if(horizontal!==null)svg+=`<line x1="${x(0)}" x2="${x(1)}" y1="${y(horizontal)}" y2="${y(horizontal)}" stroke="#a1a9af" stroke-dasharray="5 5"/>`;
 if(diagonal)svg+=`<line x1="${x(0)}" x2="${x(1)}" y1="${y(0)}" y2="${y(1)}" stroke="#a1a9af" stroke-dasharray="5 5"/>`;
 for(const s of series){if(s.line!==false)svg+=`<polyline points="${s.x.map((v,i)=>x(v)+','+y(s.y[i])).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`;if(s.dots)svg+=s.x.map((v,i)=>`<circle cx="${x(v)}" cy="${y(s.y[i])}" r="4" fill="${s.color}"/>`).join('');}
 for(const bin of intervals)svg+=`<path d="M${x(bin.mean_score)} ${y(bin.fraction_95ci[0])} V${y(bin.fraction_95ci[1])}" stroke="#8dd6ca"/><circle cx="${x(bin.mean_score)}" cy="${y(bin.candidate_fraction)}" r="4" fill="#8dd6ca"><title>${bin.count} stars; candidate fraction ${pct(bin.candidate_fraction)}</title></circle>`;
 container.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(yLabel+' versus '+xLabel)}"><g aria-hidden="true">${svg}<text x="${w/2}" y="${h-4}" text-anchor="middle">${escapeHTML(xLabel)}</text><text x="${p.l}" y="11">${escapeHTML(yLabel)}</text></g></svg>`;
}
function renderDiagnostics(){
 if(!diagnosticsData)return;const d=diagnosticsData;
 unitChart($('#pr-chart'),[{...d.models.cnn.pr,x:d.models.cnn.pr.recall,y:d.models.cnn.pr.precision,color:'#8dd6ca'},{x:d.models.baseline.pr.recall,y:d.models.baseline.pr.precision,color:'#f1bb73'}],'Recall','Precision',{horizontal:d.prevalence});
 unitChart($('#calibration-chart'),[],'Mean CNN score','Observed candidate fraction',{diagonal:true,intervals:d.models.cnn.reliability});
 $('#calibration-table').innerHTML='<table><thead><tr><th>Score bin</th><th>Stars</th><th>Mean score</th><th>Candidate fraction</th></tr></thead><tbody>'+d.models.cnn.reliability.map(b=>`<tr><td>${number(b.lower,3)}–${number(b.upper,3)}</td><td>${b.count}</td><td>${number(b.mean_score,3)}</td><td>${pct(b.candidate_fraction)}</td></tr>`).join('')+'</tbody></table>';
 $('#subgroup-table').innerHTML='<table><thead><tr><th>Subgroup</th><th>Stars</th><th>Precision</th><th>Recall</th></tr></thead><tbody>'+d.subgroups.map(g=>`<tr><td>${escapeHTML(g.name)}</td><td>${g.count}</td><td>${metricText(g.precision)}</td><td>${metricText(g.recall)}</td></tr>`).join('')+'</tbody></table>';
 $('#error-heading').textContent=`Review ${d.errors.length} test-set disagreements`;
 $('#error-table').innerHTML=d.errors.length?'<table><thead><tr><th>Target</th><th>Disagreement</th><th>CNN score</th><th>Baseline score</th><th>Period</th><th>Eclipse flag</th></tr></thead><tbody>'+d.errors.map(o=>`<tr><td><button class="text-button inspect-error" data-id="${escapeHTML(o.kepoi_name)}">${escapeHTML(o.kepoi_name)}</button></td><td>${o.error_type==='false_alarm'?'False alarm':'Missed candidate'}</td><td>${number(o.score,3)}</td><td>${number(o.baseline_score,3)}</td><td>${number(o.koi_period)} d</td><td>${o.koi_fpflag_ss?'Yes':'No'}</td></tr>`).join('')+'</tbody></table>':'<p>No disagreements at the saved threshold.</p>';
 document.querySelectorAll('.inspect-error').forEach(b=>b.onclick=()=>{$('.nav[data-page="explorer"]').click();$('#split').value='test';$('#label').value='';$('#outcome').value='';$('#search').value=b.dataset.id;renderTargets();$('#workspace').scrollIntoView({block:'start'});});
}
async function showThreshold(value){
 const ticket=++thresholdRequest;$('#threshold-value').textContent=number(value,3);$('#threshold-result').textContent='Calculating validation trade-offs…';
 try{const d=await api('/api/validation-threshold?threshold='+encodeURIComponent(value));if(ticket!==thresholdRequest)return;
 $('#threshold-result').innerHTML=`<div class="threshold-stats"><span><strong>${metricText(d.precision)}</strong>Precision</span><span><strong>${metricText(d.recall)}</strong>Recall</span><span><strong>${d.selected}/${d.count}</strong>Selected stars</span></div><p class="subtle">Validation only · ${d.confusion_matrix[0][1]} false alarms · ${d.confusion_matrix[1][0]} missed candidates</p>`;
 }catch(e){if(ticket===thresholdRequest)$('#threshold-result').textContent=e.message;}
}
async function loadDiagnostics(){
 try{diagnosticsData=await api('/api/diagnostics');renderDiagnostics();$('#threshold-slider').value=overview.metrics.threshold;showThreshold(overview.metrics.threshold);}catch(e){$('#diagnostic-status').textContent='Diagnostics unavailable: '+e.message;}
}
$('#threshold-slider').addEventListener('input',e=>{clearTimeout(thresholdTimer);thresholdRequest++;$('#threshold-value').textContent=number(e.target.value,3);$('#threshold-result').textContent='Calculating validation trade-offs…';thresholdTimer=setTimeout(()=>showThreshold(Number(e.target.value)),100);});
$('#threshold-reset').onclick=()=>{clearTimeout(thresholdTimer);$('#threshold-slider').value=overview.metrics.threshold;showThreshold(overview.metrics.threshold);};
document.querySelector('.nav[data-page="evaluation"]').addEventListener('click',renderDiagnostics);
window.addEventListener('resize',()=>{if($('#evaluation').classList.contains('active'))renderDiagnostics();});

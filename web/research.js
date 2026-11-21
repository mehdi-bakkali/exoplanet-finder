'use strict';
let researchStudy=null;
async function loadResearch(){
 try{researchStudy=await api('/api/research');renderResearch();}catch(e){$('#research-status').textContent='Research results unavailable: '+e.message;}
}
function renderResearch(){
 if(!researchStudy)return;
 const {dataset,study,injection}=researchStudy;
 $('#research-status').textContent=dataset?`${dataset.usable.toLocaleString()} usable stars · Q9 + Q10 · all pilot stars excluded · ${study?'12 fits complete':'training results pending'}`:'Acquire the multi-quarter dataset to populate this study.';
 if(study){
 $('#research-cards').innerHTML=[['Usable stars',study.dataset_size.toLocaleString()],['Training / validation',study.counts.train+' / '+study.counts.validation],['Calibration / test',study.counts.calibration+' / '+study.counts.test],['Fits / seeds','12 / 3']].map(([a,b])=>`<div class="metric-card"><span>${a}</span><strong>${b}</strong></div>`).join('');
 $('#ablation-table').innerHTML='<table><thead><tr><th>Input mode</th><th>Test AP</th><th>Test ROC AUC</th><th>Test F1</th></tr></thead><tbody>'+Object.entries(study.modes).map(([mode,v])=>`<tr><td>${escapeHTML(mode==='mask'?'Mask only':mode)}</td>${['average_precision','roc_auc','f1'].map(k=>`<td>${number(v[k].mean,3)} ± ${number(v[k].std,3)}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
 const e=study.ensemble;
 $('#research-calibration').innerHTML='<table><thead><tr><th>Ensemble</th><th>Brier ↓</th><th>Log loss ↓</th><th>AP</th></tr></thead><tbody>'+[['Raw',e.raw_test],['Calibrated',e.calibrated_test]].map(([name,v])=>`<tr><td>${name}</td><td>${number(v.brier,4)}</td><td>${number(v.log_loss,4)}</td><td>${number(v.average_precision,4)}</td></tr>`).join('')+'</tbody></table>';
 unitChart($('#research-reliability'),[],'Mean calibrated score','Observed candidate fraction',{diagonal:true,intervals:e.calibrated_test.reliability});
 }
 if(injection){
 $('#recovery-table').innerHTML='<table><thead><tr><th>Injected period</th><th>Depth</th><th>Strict recoveries</th><th>Recovery fraction</th></tr></thead><tbody>'+injection.cells.map(c=>`<tr><td>${c.injected_period} d</td><td>${c.injected_depth_ppm} ppm</td><td>${c.sum} / ${c.count}</td><td>${pct(c.mean)}</td></tr>`).join('')+'</tbody></table>';
 $('#recovery-summary').textContent=`${injection.strict_recovered}/${injection.trials} strict recoveries; ${injection.harmonic_matches} harmonic matches. ${injection.null_backgrounds_above_snr_7_1}/${injection.plan.stars} uninjected backgrounds also exceed the exploratory SNR threshold. A high BLS peak alone does not establish an injected transit recovery.`;
 }
}
$('.nav[data-page="research"]').addEventListener('click',loadResearch);
window.addEventListener('resize',()=>{if($('#research').classList.contains('active'))renderResearch();});
$('#search-bls').onclick=async()=>{
 const button=$('#search-bls'),out=$('#bls-result');button.disabled=true;out.textContent='Searching periods from 0.5 to 15 days…';
 try{
 const file=$('#curve-file').files[0];if(!file&&!exampleCSV)throw Error('Choose a CSV or load the real example first.');
 if(file?.size>8_000_000)throw Error('Use a CSV smaller than 8 MB.');
 const {time,flux}=ExoCSV.parse(file?await file.text():exampleCSV);
 const result=await api('/api/bls',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({time,flux,minimum_period:.5,maximum_period:15})});
 $('#period').value=result.period;$('#epoch').value=result.epoch;$('#duration').value=result.duration_hours;
 out.textContent=`Strongest box match: ${number(result.period,5)} d; depth SNR ${number(result.depth_snr,1)}. Ephemeris filled below. Inspect it before running the CNN. ${result.warning}`;
 }catch(e){out.textContent=e.message;}finally{button.disabled=false;}
};

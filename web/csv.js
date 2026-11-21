/* Pure CSV parser shared by browser inference and Node tests. */
(function(root){
 function cells(line){
  const out=[];let value='',quoted=false,closed=false;
  for(let i=0;i<line.length;i++){
   const c=line[i];
   if(quoted){if(c==='"'){if(line[i+1]==='"'){value+='"';i++;}else{quoted=false;closed=true;}}else value+=c;}
   else if(c===','){out.push(value.trim());value='';closed=false;}
   else if(c==='"'){if(value.trim()||closed)throw Error('Malformed CSV quotation.');quoted=true;value='';}
   else{if(closed&&c.trim())throw Error('Unexpected text after a quoted CSV field.');value+=c;}
  }
  if(quoted)throw Error('Unclosed CSV quotation. Multiline fields are not supported.');
  out.push(value.trim());return out;
 }
 function parse(text){
  const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);
  const header=cells(lines.shift()||'').map(v=>v.toLowerCase());
  if(header.filter(v=>v==='time').length!==1||header.filter(v=>v==='flux').length!==1)throw Error('CSV must have exactly one time and one flux column.');
  const ti=header.indexOf('time'),fi=header.indexOf('flux'),time=[],flux=[];
  for(let i=0;i<lines.length;i++){
   if(!lines[i].trim())continue;
   const row=cells(lines[i]);
   if(row.length!==header.length)throw Error(`CSV row ${i+2} has a different number of columns.`);
   if(row[ti]===''||row[fi]==='')throw Error(`CSV row ${i+2} has an empty time or flux.`);
   const t=Number(row[ti]),f=Number(row[fi]);
   if(!Number.isFinite(t)||!Number.isFinite(f))throw Error(`CSV row ${i+2} must contain finite numbers.`);
   time.push(t);flux.push(f);
   if(time.length>100000)throw Error('Use at most 100,000 observations.');
  }
  if(time.length<200)throw Error('Use at least 200 observations.');
  return {time,flux};
 }
 root.ExoCSV={parse};
 if(typeof module!=='undefined'&&module.exports)module.exports=root.ExoCSV;
})(globalThis);

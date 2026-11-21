const {test}=require('node:test');
const assert=require('node:assert/strict');
const {parse}=require('../web/csv.js');
const rows=Array.from({length:200},(_,i)=>`${i},${1+i/10000}`).join('\n');
test('numeric CSV supports BOM, CRLF and quoted column names',()=>{
 const data=parse('\uFEFF"time","flux"\r\n'+rows.replaceAll('\n','\r\n'));
 assert.equal(data.time.length,200);assert.equal(data.time[199],199);assert.equal(data.flux[0],1);
});
test('rejects ambiguous headers, missing cells and nonfinite values',()=>{
 for(const text of ['time,time,flux\n'+rows,'time,flux\n0,\n'+rows,'time,flux\n0,NaN\n'+rows,'time,flux\n0,Infinity\n'+rows,'time,flux\n0\n'+rows])assert.throws(()=>parse(text));
});
test('rejects malformed quotations and insufficient observations',()=>{
 assert.throws(()=>parse('time,flux\n"0,1\n'+rows));assert.throws(()=>parse('time,flux\n0,1'));
});
test('supports quoted numeric scientific notation',()=>{
 const data=parse('flux,time\n'+Array.from({length:200},(_,i)=>`"1.2e5","${i}"`).join('\n'));
 assert.equal(data.flux[0],120000);assert.equal(data.time[199],199);
});

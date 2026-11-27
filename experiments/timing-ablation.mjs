import fs from "node:fs";
import path from "node:path";
// Small matrix operations with reverse-mode gradients for the two attention paths.
const W = 8;
const names = ["track","search","wait"];
const zeros = (r,c) => Array.from({length:r},()=>Array(c).fill(0));
const node = a => ({a, g:zeros(a.length,a[0].length)});
function matrixModel(oriented, seed) {
  function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
  const params=[];
  function param(r,c){const p=node(zeros(r,c).map(row=>row.map(()=> (random()-.5)*1.6/Math.sqrt(r))));
    params.push(p);return p}
  const symbol=param(3,W), age=param(6,W), option=param(3,W);
  const oq=param(W,W), ok=param(W,W), ov=param(W,W);
  const cq=param(W,W), ck=param(W,W), cv=param(W,W), read=param(W,1);
  function forward(window, target=null, lr=.014){
    const back=[];
    function mul(x,y){
      const z=node(zeros(x.a.length,y.a[0].length));
      for(let i=0;i<x.a.length;i++)for(let k=0;k<y.a.length;k++)for(let j=0;j<z.a[0].length;j++)
        z.a[i][j]+=x.a[i][k]*y.a[k][j];
      back.push(()=>{for(let i=0;i<x.a.length;i++)for(let k=0;k<y.a.length;k++)for(let j=0;j<z.a[0].length;j++){
        x.g[i][k]+=z.g[i][j]*y.a[k][j]; y.g[k][j]+=z.g[i][j]*x.a[i][k]}});
      return z;
    }
    function mix(x,y){
      const z=node(x.a.map((row,i)=>row.map((v,j)=>Math.tanh(v+y.a[i][j]))));
      back.push(()=>z.g.forEach((row,i)=>row.forEach((g,j)=>{
        const d=g*(1-z.a[i][j]**2); x.g[i][j]+=d;y.g[i][j]+=d})));return z;
    }
    function attention(q,k,v){
      const weights=q.a.map(row=>softmax(k.a.map(key=>dot(row,key)/Math.sqrt(W))));
      const att=node(weights);
      back.push(()=>weights.forEach((row,i)=>{
        const avg=dot(att.g[i],row);
        row.forEach((p,j)=>{const ds=p*(att.g[i][j]-avg)/Math.sqrt(W);
          for(let h=0;h<W;h++){q.g[i][h]+=ds*k.a[j][h];k.g[j][h]+=ds*q.a[i][h]}})
      }));
      return mul(att,v);
    }
    const x=node(window.map((s,i)=>symbol.a[s].map((v,j)=>v+age.a[i][j])));
    back.push(()=>window.forEach((s,i)=>x.g[i].forEach((g,j)=>{symbol.g[s][j]+=g;age.g[i][j]+=g})));
    const history=oriented?mix(x,attention(mul(x,oq),mul(x,ok),mul(x,ov))):x;
    const evidence=attention(mul(option,cq),mul(history,ck),mul(history,cv));
    const z=mul(mix(option,evidence),read), values=z.a.map(r=>r[0]);
    if(target){
      const grad=gradient(values,target);
      grad.forEach((g,i)=>z.g[i][0]=g);
      for(let i=back.length-1;i>=0;i--)back[i]();
      for(const p of params)for(let i=0;i<p.a.length;i++)for(let j=0;j<p.a[i].length;j++){
        p.a[i][j]-=lr*p.g[i][j];p.g[i][j]=0}
    }
    return values;
  }
  return forward;
}
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
function softmax(x){const e=x.map(v=>Math.exp(v-Math.max(...x))),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
function gradient(z,labels){const p=softmax(z),mass=p.reduce((s,x,i)=>s+(labels.includes(names[i])?x:0),0);
  return p.map((x,i)=>x-(labels.includes(names[i])?x/mass:0))}
const index=w=>w.reduce((v,x)=>v*3+x,0);
function allWindows(){return Array.from({length:729},(_,i)=>Array.from({length:6},(_,j)=>Math.floor(i/3**(5-j))%3))}
function ticks(sessions){
  const rows=[];
  for(const s of sessions){
    let lastValid=null,lastHit=null;
    for(let t=0;t<s.frames.length;t++){
      const now=t*200, symbol=s.frames[t];
      if(symbol){lastValid=now-(s.receive_age_ms?.[t]??0);if(symbol===2)lastHit=lastValid}
      if(t<5)continue;
      const w=s.frames.slice(t-5,t+1), accepted=s.labels?s.labels[t-5].split("|"):s.ticks[t-5].accepted;
      const time=[1,lastValid===null?0:Math.min(now-lastValid,2000)/1000,
                  lastHit===null?0:Math.min(now-lastHit,2000)/1000,Number(lastHit===null),Number(lastValid===null)];
      rows.push({id:s.id??s.session,t,w,time,accepted});
    }
  }
  return rows;
}
function fitBase(oriented,training){
  const net=matrixModel(oriented,17);
  // Deterministic stride samples different positions in every session on successive epochs.
  for(let epoch=0;epoch<24;epoch++){
    for(let i=epoch%4;i<training.length;i+=4){const r=training[i];if(r.w[5])net(r.w,r.accepted,.025)}
  }
  return allWindows().map(w=>net(w).map(Math.fround));
}
function timeHead(table,development){
  const coefficients=zeros(3,5);
  const predict=r=>table[index(r.w)].map((base,j)=>base+dot(coefficients[j],r.time));
  for(let epoch=0;epoch<60;epoch++)for(const r of development){
    if(!r.w[5])continue;
    const g=gradient(predict(r),r.accepted);
    for(let j=0;j<3;j++)for(let i=0;i<5;i++)coefficients[j][i]-=.035*(g[j]*r.time[i]+.005*coefficients[j][i]);
  }
  return {predict,coefficients};
}
function timedMlp(training,development){
  let seed=17;
  const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed/4294967296-.5)*.5};
  const w=Array.from({length:16},()=>Array.from({length:10},rand)),v=Array.from({length:3},()=>Array.from({length:16},rand));
  function evalRow(r,rate=0){
    const x=[1,Number(r.w[5]===2),r.w.filter(s=>s===2).length/6,r.w.filter(s=>s===1).length/6,
             r.w.slice(-3).filter(s=>s===2).length/3,...r.time];
    const h=w.map(row=>Math.tanh(dot(row,x))),z=v.map(row=>dot(row,h));
    if(rate){
      const g=gradient(z,r.accepted);
      for(let j=0;j<h.length;j++){
        const d=(1-h[j]*h[j])*g.reduce((s,e,k)=>s+e*v[k][j],0);
        for(let i=0;i<x.length;i++)w[j][i]-=rate*d*x[i];
        for(let k=0;k<3;k++)v[k][j]-=rate*g[k]*h[j];
      }
    }
    return z;
  }
  for(let e=0;e<8;e++)for(const r of training)if(r.w[5])evalRow(r,.018);
  for(let e=0;e<30;e++)for(const r of development)if(r.w[5])evalRow(r,.012);
  return evalRow;
}
function run(train,dev,test,production){
  const training=ticks(train),development=ticks(dev),holdout=ticks(test);
  const a=fitBase(false,training),b=fitBase(true,training);
  const at=timeHead(a,development),bt=timeHead(b,development),pt=timeHead(production,development);
  const choice=z=>names[z.indexOf(Math.max(...z))];
  const mlp=timedMlp(training,development);
  const methods={timed_mlp:r=>choice(mlp(r)),A:r=>choice(a[index(r.w)]),A_time:r=>choice(at.predict(r)),
                 B:r=>choice(b[index(r.w)]),B_time:r=>choice(bt.predict(r)),
                 production:r=>choice(production[index(r.w)]),production_time:r=>choice(pt.predict(r)),
                 timed_rule:r=>r.w[5]===2?(r.w.slice(-3).filter(v=>v===2).length>=2?"track":"wait"):
                   (r.time[3]||r.time[2]>=.65?"search":"wait")};
  const detail=holdout.map(r=>({session:r.id,tick:r.t,accepted:r.accepted,
    predictions:Object.fromEntries(Object.entries(methods).map(([name,method])=>[name,r.w[5]?method(r):"defer"]))}));
  const counts=Object.fromEntries(Object.keys(methods).map(k=>[k,detail.filter(r=>r.accepted.includes(r.predictions[k])).length]));
  return {a,b,methods,report:{ticks:detail.length,counts,coefficients:{A:at.coefficients,B:bt.coefficients,production:pt.coefficients},detail}};
}

const read=p=>fs.readFileSync(new URL('../'+p,import.meta.url),'utf8').trim();
const lines=p=>read(p).split('\n').map(JSON.parse);
const production=read("edge/scores.inc").split('\n').map(l=>l.match(/[-+]?\d+\.\d+e[-+]\d+/gi).map(Number));
const {b,methods,report}=run(lines("data/train.jsonl"),lines("data/timing-development.jsonl"),lines("data/timing-holdout.jsonl"),production);
const out=process.argv[2];
if(!out)throw Error("Usage: node experiments/timing-ablation.mjs OUTPUT_DIRECTORY");
fs.mkdirSync(out,{recursive:true});
const data=Buffer.alloc(729*3*4);
b.flat().forEach((v,i)=>data.writeFloatLE(v,i*4));
fs.writeFileSync(path.join(out,"B.f32le"),data);
fs.writeFileSync(path.join(out,"timing-ablation.json"),JSON.stringify(report,null,2)+"\n");
console.log(report.counts);

if (process.argv[3]) {
  // Reviewer annotations are joined by (session,tick) after proposals have been captured.
  const records = fs.readFileSync(process.argv[3],"utf8").trim().split('\n').map(JSON.parse);
  const annotations = process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4],"utf8")) : {};
  let last = null, window = [];
  const field = records.map(row=>{
    if (!Object.hasOwn(row,"positive_age_ms") || !Object.hasOwn(row,"valid_age_ms"))
      throw Error("Field comparison needs the timestamped trace format.");
    if (!last || last.session!==row.session || last.tick+1!==row.tick ||
        last.source!==row.source || last.eligible!==row.eligible) window=[];
    window=[...window,row.symbol].slice(-6);last=row;
    const scale=ms=>ms===null?0:Math.min(ms,2000)/1000;
    const time=[1,scale(row.valid_age_ms),scale(row.positive_age_ms),
                Number(row.positive_age_ms===null),Number(row.valid_age_ms===null)];
    const active=window.length===6 && row.eligible && row.symbol!==0;
    if (active && row.mask!==7) throw Error("Field comparison currently requires the three-candidate shadow contract.");
    const predictions=Object.fromEntries(Object.entries(methods).map(([key,fn])=>
        [key,active?fn({w:window,time}):"defer"]));
    if (predictions.production!==row.proposal) throw Error("Production mismatch at "+row.session+":"+row.tick);
    return {session:row.session,tick:row.tick,incumbent:row.incumbent,accepted:annotations[row.session]?.[row.tick]??null,predictions};
  });
  fs.writeFileSync(path.join(out,"field-replay.jsonl"),field.map(row=>JSON.stringify(row)).join('\n')+'\n');
}

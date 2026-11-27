import fs from 'node:fs';
const names = ['track', 'search', 'wait'];
const dot = (a,b) => a.reduce((s,v,i) => s + v*b[i], 0);
const zeros = n => Array(n).fill(0);
const best = a => a.indexOf(Math.max(...a));
const count = (w,n) => w.filter(x => x===n).length;
function signals(w) {
  const last = w.lastIndexOf(2);
  const elapsed = last<0 ? 6 : 5-last;
  const negatives = count(w.slice(last+1),1);
  return {elapsed, negatives, recent:count(w.slice(-3),2)};
}
function incumbent(w, previous) {
  const {elapsed,recent} = signals(w);
  if (!w[5]) return 'defer';
  if (!elapsed) return recent>=2 ? 'track' : 'wait';
  if (elapsed===3 && previous==='search') return 'search';
  return elapsed>=4 ? 'search' : 'wait';
}
function samples(sessions, table) {
  const rows=[];
  for (const s of sessions) {
    let previous='unknown';
    for(let t=5;t<s.frames.length;t++) {
      const w=s.frames.slice(t-5,t+1);
      const {elapsed,negatives,recent}=signals(w);
      const rule=incumbent(w,previous);
      const features=[1,elapsed/6,negatives/6,recent/3,w[5]===2?1:0,count(w,0)/6,
                      ...names.map(n=>Number(previous===n))];
      const index=w.reduce((a,x)=>a*3+x,0);
      const allowed=s.labels ? s.labels[t-5].split('|') : s.ticks[t-5].accepted;
      rows.push({id:s.id??s.session,t,rule,w,features,base:table[index],allowed,
                 mlpInput:[Number(w[5]===2),elapsed/6,count(w,2)/6,1-count(w,0)/6]});
      previous=rule; // Offline incumbent proxy. Never use the current label as input.
    }
  }
  return rows;
}
function gradient(scores, allowed) {
  const exp=scores.map(s=>Math.exp(s-Math.max(...scores)));
  const total=exp.reduce((a,b)=>a+b);
  const mass=exp.reduce((a,b,i)=>a+(allowed.includes(names[i])?b:0),0);
  return exp.map((v,i)=>v/total-(allowed.includes(names[i])?v/mass:0));
}
function fitSummary(rows) {
  // Shared 6 -> 16 -> 1 MLP, trained here independently of the historical PyTorch run.
  let seed=17;
  function init(){seed=(Math.imul(1664525,seed)+1013904223)>>>0;return (seed/4294967296-.5)*.7}
  const W=Array.from({length:16},()=>Array.from({length:7},init)), v=Array.from({length:16},init);
  function evaluate(row, train=false) {
    const inputs=[[1,0],[0,1],[0,0]].map(o=>[...row.mlpInput,...o,1]);
    const hidden=inputs.map(x=>W.map(weights=>Math.tanh(dot(x,weights))));
    const scores=hidden.map(h=>dot(h,v));
    if(train) {
      const g=gradient(scores,row.allowed);
      for(let j=0;j<16;j++){
        const old=v[j];
        for(let k=0;k<3;k++){
          const h=hidden[k][j], step=.025*g[k];
          v[j]-=step*h;
          for(let i=0;i<7;i++) W[j][i]-=step*old*(1-h*h)*inputs[k][i];
        }
      }
    }
    return scores;
  }
  for(let e=0;e<10;e++) for(const row of rows) if(row.w[5]) evaluate(row,true);
  return evaluate;
}
function experiment(train, development, holdout, table) {
  const mlp=fitSummary(samples(train,table));
  const rows=samples(development,table);
  const weights=names.map(()=>zeros(9));
  // Fixed base logit scale prevents a high-confidence synthetic teacher from drowning out a small residual.
  const score=r=>r.base.map((v,j)=>v*.25+dot(weights[j],r.features));
  for(let e=0;e<35;e++) for(const r of rows) if(r.w[5]){
    const g=gradient(score(r),r.allowed);
    for(let j=0;j<3;j++) for(let i=0;i<9;i++)
      weights[j][i]-=.04*(g[j]*r.features[i]+.003*weights[j][i]);
  }
  function evaluate(set){
    const detail=samples(set,table).map(r=>{
      const eligible=r.w[5]!==0;
      const pick=z=>eligible?names[best(z)]:'defer';
      const result={rule:r.rule,mlp:pick(mlp(r)),transformer:pick(r.base),context:pick(score(r))};
      return {id:r.id,tick:r.t,accepted:r.allowed,...result};
    });
    return {ticks:detail.length,accepted:Object.fromEntries(['rule','mlp','transformer','context'].map(k=>
      [k,detail.filter(r=>r.accepted.includes(r[k])).length])),detail};
  }
  return {summary_mlp_seed:17,context_weights:weights,development:evaluate(development),holdout:evaluate(holdout)};
}
const read = p => fs.readFileSync(new URL('../'+p,import.meta.url),'utf8').trim();
const rows = p => read(p).split('\n').map(JSON.parse);
const table = read('edge/scores.inc').split('\n').map(line=>line.match(/[-+]?\d+\.\d+e[-+]\d+/gi).map(Number));
console.log(JSON.stringify(experiment(rows('data/train.jsonl'),rows('data/review-cases.jsonl'),
                                     rows('data/context-holdout.jsonl'),table),null,2));

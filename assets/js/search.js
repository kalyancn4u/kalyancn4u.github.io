class SimpleSearch{
constructor(){this.i=document.getElementById('search-input');this.r=document.getElementById('search-results');fetch('/search.json').then(r=>r.json()).then(d=>{this.docs=d;this.i.addEventListener('input',()=>this.run());});}
n(s){return (s||'').toLowerCase();}
score(d,q){q=this.n(q);let s=0;[['title',10],['aliases',8],['keywords',7],['tags',5],['categories',5],['content',1]].forEach(f=>{let t=this.n(d[f[0]]);if(t.includes(q))s+=f[1];});return s;}
run(){let q=this.i.value.trim();this.r.innerHTML='';if(!q)return;this.docs.map(d=>({d,s:this.score(d,q)})).filter(x=>x.s>0).sort((a,b)=>b.s-a.s).slice(0,20).forEach(x=>{let e=document.createElement('div');e.innerHTML=`<a href="${x.d.url}">${x.d.title}</a>`;this.r.appendChild(e);});}}
window.addEventListener('DOMContentLoaded',()=>new SimpleSearch());

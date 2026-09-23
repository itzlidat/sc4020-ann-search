import os, json, time
import numpy as np
from collections import defaultdict

# ---------------- Configuration ----------------
SIFT_BASE='sift_base.fvecs'; SIFT_QUERY='sift_query.fvecs'; SIFT_GT='sift_groundtruth.ivecs'
WIKI_BASE='wiki_base_embeddings.npy'; WIKI_QUERY='wiki_query_embeddings.npy'; WIKI_TRAIN='wiki_train_embeddings.npy'; WIKI_SENT='wiki_sentences.json'
MAX_QUERIES=None; K_VALUES=[1,10,100]
RANDOM_SEED=42; BUILD_CHUNK_SIZE=100_000; MAX_CANDIDATES=None
MAIN=(20,4,800.0)
SIFT_REPS=[('SIFT 20x4 w=400',20,4,400.0),('SIFT 20x4 w=800',20,4,800.0),('SIFT 40x4 w=800',40,4,800.0)]
WIKI_REPS=[('Wiki 10x4 w=0.5',10,4,0.5),('Wiki 10x4 w=1.0',10,4,1.0),('Wiki 20x4 w=1.5',20,4,1.5),('Wiki 20x4 w=2.0',20,4,2.0),('Wiki 40x4 w=2.0',40,4,2.0)]
RUN_REPRESENTATIVES=True

# ---------------- I/O ----------------
def read_fvecs(path):
    with open(path,'rb') as f:
        dim=int(np.fromfile(f,np.int32,1)[0]); f.seek(0,2); size=f.tell(); stride=4+4*dim
        if size%stride: raise ValueError(f'Invalid .fvecs file: {path}')
        n=size//stride; f.seek(0); a=np.fromfile(f,np.float32).reshape(n,dim+1)
    return np.ascontiguousarray(a[:,1:])

def read_ivecs(path):
    with open(path,'rb') as f:
        dim=int(np.fromfile(f,np.int32,1)[0]); f.seek(0); a=np.fromfile(f,np.int32)
    stride=dim+1
    if len(a)%stride: raise ValueError(f'Invalid .ivecs file: {path}')
    return np.ascontiguousarray(a.reshape(-1,stride)[:,1:])

def norms(x): return np.sum(x*x,axis=1,dtype=np.float32)
def dist(q,x,n): return np.maximum(np.dot(q,q)+n-2*np.dot(x,q),0.0)
def topk(q,x,n,k):
    d=dist(q,x,n)
    if k>=len(d): return np.argsort(d)
    ids=np.argpartition(d,k-1)[:k]; return ids[np.argsort(d[ids])]
def recall(pred,gt,k): return len(set(map(int,pred[:k]))&set(map(int,gt[:k])))/k

def exact_gt(db,q,dn):
    gt=np.empty((len(q),max(K_VALUES)),np.int32); t=time.perf_counter()
    for i,x in enumerate(q): gt[i]=topk(x,db,dn,max(K_VALUES))
    print(f'Ground truth: {time.perf_counter()-t:.2f} s'); return gt

# ---------------- LSH ----------------
class RandomProjectionLSH:
    def __init__(self,dim,tables,hashes,width,seed=42,chunk=100_000):
        self.tables_n=tables; self.hashes=hashes; self.width=float(width); self.chunk=chunk
        rng=np.random.default_rng(seed)
        self.proj=rng.normal(size=(tables,hashes,dim)).astype(np.float32)
        self.offset=rng.uniform(0,self.width,size=(tables,hashes)).astype(np.float32)
        self.tables=[]
    def fit(self,db):
        self.tables=[]; t=time.perf_counter()
        for ti in range(self.tables_n):
            tab=defaultdict(list); p=self.proj[ti]; o=self.offset[ti]
            for s in range(0,len(db),self.chunk):
                e=min(s+self.chunk,len(db)); h=np.floor((np.dot(db[s:e],p.T)+o)/self.width).astype(np.int64)
                for j,row in enumerate(h): tab[tuple(row)].append(s+j)
            self.tables.append(tab)
            print(f'  table {ti+1}/{self.tables_n}: {len(tab):,} buckets')
        return time.perf_counter()-t
    def candidates(self,q):
        ids=set()
        for ti in range(self.tables_n):
            h=np.floor((np.dot(self.proj[ti],q)+self.offset[ti])/self.width).astype(np.int64)
            ids.update(self.tables[ti].get(tuple(h),()))
        return np.fromiter(ids,dtype=np.int32) if ids else np.empty(0,np.int32)
    def search(self,q,db,dn,k):
        ids=self.candidates(q); retrieved=len(ids)
        if MAX_CANDIDATES is not None and retrieved>MAX_CANDIDATES: ids=ids[:MAX_CANDIDATES]
        if len(ids)==0: return ids,retrieved
        d=dist(q,db[ids],dn[ids]); kk=min(k,len(ids)); part=np.argpartition(d,kk-1)[:kk]; part=part[np.argsort(d[part])]
        return ids[part],len(ids)

# ---------------- Evaluation ----------------
def linear_baseline(db,q,dn,gt):
    t=time.perf_counter(); sums={k:0.0 for k in K_VALUES}
    for i,x in enumerate(q):
        p=topk(x,db,dn,max(K_VALUES))
        for k in K_VALUES: sums[k]+=recall(p,gt[i],k)
    e=time.perf_counter()-t
    return {'query_ms':e/len(q)*1000,'qps':len(q)/e,'recall':{k:sums[k]/len(q) for k in K_VALUES}}

def run_config(db,q,dn,gt,name,tables,hashes,width):
    print(f'\n{name}: {tables} tables x {hashes} hashes, width={width}')
    lsh=RandomProjectionLSH(db.shape[1],tables,hashes,width,RANDOM_SEED,BUILD_CHUNK_SIZE)
    build=lsh.fit(db); sums={k:0.0 for k in K_VALUES}; counts=[]; t=time.perf_counter()
    for i,x in enumerate(q):
        p,c=lsh.search(x,db,dn,max(K_VALUES)); counts.append(c)
        for k in K_VALUES: sums[k]+=recall(p,gt[i],k)
        if (i+1)%1000==0: print(f'  {i+1:,}/{len(q):,}')
    e=time.perf_counter()-t; counts=np.asarray(counts); r={k:sums[k]/len(q) for k in K_VALUES}
    return {'name':name,'tables':tables,'hashes':hashes,'width':width,'build':build,'query_ms':e/len(q)*1000,'qps':len(q)/e,'candidates':float(counts.mean()),'db_pct':100*counts.mean()/len(db),'recall':r}

def run_dataset(name,db,q,gt,reps):
    dn=norms(db); linear=linear_baseline(db,q,dn,gt); configs=[('Main LSH configuration',*MAIN)]
    if RUN_REPRESENTATIVES:
        configs += [(n,t,h,w) for n,t,h,w in reps if (t,h,w)!=MAIN]
    results=[]
    for c in configs: results.append(run_config(db,q,dn,gt,*c))
    print('\n'+name+' SUMMARY')
    print(f'{'Configuration':30} {'Query(ms)':>11} {'QPS':>10} {'Speedup':>9} {'Candidates':>12} {'DB%':>9} {'R@1':>8} {'R@10':>8} {'R@100':>8}')
    for r in results:
        sp=linear['query_ms']/r['query_ms']; print(f"{r['name'][:30]:30} {r['query_ms']:11.4f} {r['qps']:10.2f} {sp:9.2f} {r['candidates']:12,.0f} {r['db_pct']:9.4f} {r['recall'][1]:8.4f} {r['recall'][10]:8.4f} {r['recall'][100]:8.4f}")
    print(f"{'Exact linear scan':30} {linear['query_ms']:11.4f} {linear['qps']:10.2f} {1.0:9.2f} {'-':>12} {100:9.4f} {linear['recall'][1]:8.4f} {linear['recall'][10]:8.4f} {linear['recall'][100]:8.4f}")
    return linear,results

def run_sift():
    for p in [SIFT_BASE,SIFT_QUERY,SIFT_GT]:
        if not os.path.exists(p): raise FileNotFoundError(p)
    db=read_fvecs(SIFT_BASE); q=read_fvecs(SIFT_QUERY); gt=read_ivecs(SIFT_GT)
    if MAX_QUERIES is not None: q=q[:min(MAX_QUERIES,len(q))]; gt=gt[:len(q)]
    print(f'\nSIFT: {db.shape}, queries={len(q):,}'); return run_dataset('SIFT1M',db,q,gt,SIFT_REPS)

def run_wiki():
    db=np.load(WIKI_BASE,allow_pickle=False); q=np.load(WIKI_QUERY,allow_pickle=False); train=np.load(WIKI_TRAIN,allow_pickle=False)
    try:
        with open(WIKI_SENT,encoding='utf8') as f: meta=json.load(f)
        print(f'Wikipedia metadata entries: {len(meta):,}')
    except FileNotFoundError: pass
    if MAX_QUERIES is not None: q=q[:min(MAX_QUERIES,len(q))]
    dn=norms(db); gt=exact_gt(db,q,dn); print(f'\nWikipedia: {db.shape}, queries={len(q):,}, train={train.shape}')
    return run_dataset('Wikipedia',db,q,gt,WIKI_REPS)

def main():
    print('='*100); print('RANDOM PROJECTION LSH — SIFT1M + WIKIPEDIA'); print('='*100)
    s=run_sift(); w=run_wiki()
    print('\nExperiment complete.')
if __name__=='__main__': main()

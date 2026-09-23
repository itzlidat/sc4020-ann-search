import os, json, time
import numpy as np

#---------------- Configuration ----------------
SIFT_BASE='sift_base.fvecs'; SIFT_QUERY='sift_query.fvecs'; SIFT_GT='sift_groundtruth.ivecs'
WIKI_BASE='wiki_base_embeddings.npy'; WIKI_QUERY='wiki_query_embeddings.npy'; WIKI_TRAIN='wiki_train_embeddings.npy'; WIKI_SENT='wiki_sentences.json'
MAX_QUERIES=None
K_VALUES=[1,10,100]

# ---------------- SIFT I/O ----------------
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

#---------------- Common search ----------------
def norms(x): return np.sum(x*x,axis=1,dtype=np.float32)

def distances(q,x,xnorm):
    return np.maximum(np.dot(q,q)+xnorm-2.0*np.dot(x,q),0.0)

def topk(q,x,xnorm,k):
    d=distances(q,x,xnorm)
    if k>=len(d): return np.argsort(d)
    ids=np.argpartition(d,k-1)[:k]
    return ids[np.argsort(d[ids])]

def recall(pred,gt,k):
    return len(set(map(int,pred[:k])) & set(map(int,gt[:k])))/k

def exact_ground_truth(db,queries,dbnorm):
    gt=np.empty((len(queries),max(K_VALUES)),np.int32); t=time.perf_counter()
    for i,q in enumerate(queries):
        gt[i]=topk(q,db,dbnorm,max(K_VALUES))
    print(f'Ground truth: {time.perf_counter()-t:.2f} s')
    return gt

def evaluate(db,queries,dbnorm,gt,name):
    print(f'\n{name} — EXACT LINEAR SCAN')
    sums={k:0.0 for k in K_VALUES}; t=time.perf_counter()
    for i,q in enumerate(queries):
        p=topk(q,db,dbnorm,max(K_VALUES))
        for k in K_VALUES: sums[k]+=recall(p,gt[i],k)
        if (i+1)%1000==0: print(f'  {i+1:,}/{len(queries):,}')
    elapsed=time.perf_counter()-t
    r={k:sums[k]/len(queries) for k in K_VALUES}
    out={'query_ms':elapsed/len(queries)*1000,'qps':len(queries)/elapsed,'recall':r}
    print(f'  Query: {out["query_ms"]:.4f} ms | QPS: {out["qps"]:.2f} | R@1/10/100: {r[1]:.4f}/{r[10]:.4f}/{r[100]:.4f}')
    return out

def limit_queries(q,gt=None):
    if MAX_QUERIES is None: return q,gt
    n=min(MAX_QUERIES,len(q)); return q[:n], None if gt is None else gt[:n]

#---------------- Dataset runners ----------------
def run_sift():
    print('\n'+'='*80+'\nSIFT1M\n'+'='*80)
    for p in [SIFT_BASE,SIFT_QUERY,SIFT_GT]:
        if not os.path.exists(p): raise FileNotFoundError(p)
    db=read_fvecs(SIFT_BASE); q=read_fvecs(SIFT_QUERY); gt=read_ivecs(SIFT_GT)
    if db.shape[1]!=q.shape[1] or len(q)!=len(gt): raise ValueError('SIFT shapes/counts do not match')
    q,gt=limit_queries(q,gt); dn=norms(db)
    print(f'Database: {db.shape}; Queries: {q.shape}; GT: {gt.shape}')
    return evaluate(db,q,dn,gt,'SIFT1M')

def run_wiki():
    print('\n'+'='*80+'\nWIKIPEDIA SENTENCE EMBEDDINGS\n'+'='*80)
    db=np.load(WIKI_BASE,allow_pickle=False); q=np.load(WIKI_QUERY,allow_pickle=False); train=np.load(WIKI_TRAIN,allow_pickle=False)
    try:
        with open(WIKI_SENT,encoding='utf8') as f: meta=json.load(f)
        print(f'Sentence metadata entries: {len(meta):,}' if hasattr(meta,'__len__') else 'Sentence metadata loaded')
    except FileNotFoundError: print('Sentence metadata not found (optional)')
    if db.ndim!=2 or q.ndim!=2 or train.ndim!=2 or db.shape[1]!=q.shape[1] or db.shape[1]!=train.shape[1]: raise ValueError('Wikipedia embedding shapes do not match')
    q,_=limit_queries(q); dn=norms(db); gt=exact_ground_truth(db,q,dn)
    print(f'Base: {db.shape}; Queries: {q.shape}; Train: {train.shape}')
    return evaluate(db,q,dn,gt,'Wikipedia')

def main():
    print('EXACT LINEAR SCAN — SIFT1M + WIKIPEDIA')
    s=run_sift(); w=run_wiki()
    print('\n'+'='*80+'\nFINAL SUMMARY\n'+'='*80)
    for name,r in [('SIFT1M',s),('Wikipedia',w)]:
        print(f'{name:12} | {r["query_ms"]:10.4f} ms/query | {r["qps"]:10.2f} QPS | R@1 {r["recall"][1]:.4f} | R@10 {r["recall"][10]:.4f} | R@100 {r["recall"][100]:.4f}')

if __name__=='__main__': main()

"""Fixed, study-weighted growth models used in the manuscript."""
from collections import Counter, defaultdict
import math, os
def mean(x):return sum(x)/len(x)

def npmodule():
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    import numpy as np
    return np

def bysource(rows):
    out=defaultdict(list)
    for i,r in enumerate(rows):out[r['source_id']].append(i)
    return out

def msource(rows):return {s:mean([float(rows[i]['initial_weight_g']) for i in ids]) for s,ids in bysource(rows).items()}

def weights(rows):
    counts=Counter(r['source_id'] for r in rows)
    return npmodule().array([1/len(counts)/counts[r['source_id']] for r in rows])

def feature_raw(rows,withT,context=False):
    np=npmodule()
    return np.array([[math.log(float(r['initial_weight_g'])),float(r['duration_d'])]+
                     ([float(r['temperature_descriptor_C'])] if withT else [])+
                     ([] if context else [float(r['diet_protein_DM_pct']),float(r['diet_lipid_DM_pct'])]) for r in rows])

def ridge(z,y,w):
    np=npmodule();mu=w@z;ym=float(w@y);c=z-mu
    A=c.T@(w[:,None]*c)+np.eye(z.shape[1]);b=c.T@(w*(y-ym));beta=np.linalg.solve(A,b)
    assert np.allclose(A@beta,b,atol=1e-9,rtol=1e-8)
    return dict(center=mu.tolist(),intercept=ym,beta=beta.tolist())

def ridge_predict(f,z):
    np=npmodule();return f['intercept']+(z-np.array(f['center']))@np.array(f['beta'])

def transform_fit(rows,withT,context,quadratic):
    np=npmodule();x=feature_raw(rows,withT,context);w=weights(rows);mu=w@x;sd=np.sqrt(w@((x-mu)**2));active=sd>=1e-12;sd[~active]=1
    z=(x-mu)/sd;z[:,~active]=0
    f=dict(withT=withT,context=context,quadratic=quadratic,center=mu.tolist(),scale=sd.tolist(),active=active.tolist())
    if quadratic:
        sq=z[:,-2:]**2;sm=w@sq;ss=np.sqrt(w@((sq-sm)**2));sa=ss>=1e-12;ss[~sa]=1
        zs=(sq-sm)/ss;zs[:,~sa]=0;z=np.c_[z,zs]
        f.update(square_center=sm.tolist(),square_scale=ss.tolist(),square_active=sa.tolist())
    return f,z

def transform_apply(f,rows):
    np=npmodule();z=(feature_raw(rows,f['withT'],f['context'])-np.array(f['center']))/np.array(f['scale']);z[:,~np.array(f['active'])]=0
    if f['quadratic']:
        zs=(z[:,-2:]**2-np.array(f['square_center']))/np.array(f['square_scale']);zs[:,~np.array(f['square_active'])]=0;z=np.c_[z,zs]
    return z

def gain_fit(train,withT,kind='pooled',quadratic=False,context=False):
    np=npmodule();tr,z=transform_fit(train,withT,context,quadratic);w=weights(train);m=msource(train)
    y=np.array([(float(r['final_weight_g'])-float(r['initial_weight_g']))/m[r['source_id']] for r in train])
    f=dict(transform=tr,kind=kind,training_sources=sorted(m),alpha=1)
    if kind=='pooled':f['fit']=ridge(z,y,w)
    else:
        zc=z.copy();yc=y.copy();zm=[];ym=[]
        for s,ids in sorted(bysource(train).items()):
            zs=z[ids].mean(axis=0);ys=float(y[ids].mean());zc[ids]-=zs;yc[ids]-=ys;zm.append(zs);ym.append(ys)
        A=zc.T@(w[:,None]*zc)+np.eye(z.shape[1]);b=zc.T@(w*yc);beta=np.linalg.solve(A,b)
        assert np.allclose(A@beta,b,atol=1e-9,rtol=1e-8)
        f['within_beta']=beta.tolist();f['level_fit']=ridge(np.array(zm),np.array(ym),np.ones(len(zm))/len(zm))
    return f

def gain_predict(f,test,lam=1.):
    np=npmodule();z=transform_apply(f['transform'],test);m=msource(test)
    if f['kind']=='pooled':u=ridge_predict(f['fit'],z)
    else:
        u=np.zeros(len(test))
        for s,ids in bysource(test).items():
            zm=z[ids].mean(axis=0);level=float(ridge_predict(f['level_fit'],zm[None,:])[0])
            u[ids]=level+lam*(z[ids]-zm)@np.array(f['within_beta'])
    return np.array([u[i]*m[r['source_id']] for i,r in enumerate(test)])

def core_fit(train):
    np=npmodule();w=weights(train)
    x=np.array([[math.log(float(r['initial_weight_g'])),math.log(float(r['duration_d']))] for r in train]);mu=w@x;sd=np.sqrt(w@((x-mu)**2));active=sd>=1e-12;sd[~active]=1
    z=(x-mu)/sd;z[:,~active]=0;y=np.array([math.log(float(r['final_weight_g'])/float(r['initial_weight_g']))/float(r['duration_d']) for r in train])
    return dict(center=mu.tolist(),scale=sd.tolist(),active=active.tolist(),fit=ridge(z,y,w),training_sources=sorted({r['source_id'] for r in train}))

def core_predict(f,test):
    np=npmodule();x=np.array([[math.log(float(r['initial_weight_g'])),math.log(float(r['duration_d']))] for r in test]);z=(x-np.array(f['center']))/np.array(f['scale']);z[:,~np.array(f['active'])]=0
    y=ridge_predict(f['fit'],z)
    return np.array([float(r['initial_weight_g'])*math.exp(float(g)*float(r['duration_d'])) for r,g in zip(test,y)])

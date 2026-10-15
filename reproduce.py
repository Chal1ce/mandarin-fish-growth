"""Refit the manuscript models using only the processed data release."""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import argparse,csv,hashlib,json,math,os

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
from scipy.stats import spearmanr
import models as m
from evaluation import evaluate
from heterogeneity import heterogeneity

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def prediction_rows(test,final,scope,model,hold):
    return [dict(scope=scope,model=model,held_out=hold,source_id=r['source_id'],treatment_id=r['treatment_id'],
        network_stress_group=r['network_stress_group'],duration_d=r['duration_d'],initial_weight_g=float(r['initial_weight_g']),
        reported_final_weight_g=float(r['final_weight_g']),reported_gain_g=float(r['final_weight_g'])-float(r['initial_weight_g']),
        predicted_final_weight_g=float(p),predicted_gain_g=float(p)-float(r['initial_weight_g'])) for r,p in zip(test,final)]

def summarize(source):
    groups=defaultdict(list)
    for r in source:groups[(r['scope'],r['model'])].append(r)
    out=[]
    for (scope,model),a in sorted(groups.items()):
        row=dict(scope=scope,model=model,studies=len(a))
        for k in ['endpoint_MAPE','endpoint_MAE_g','contrast_NMAE','direction_accuracy','response_RMS_ratio']:
            values=[float(r[k]) for r in a if r[k]!=''];row[k]=m.mean(values) if values else ''
        out.append(row)
    return out

def gain_analysis(nut,out):
    predictions=[];splits=[]
    for sample,rows in [('primary13',nut),('matched10',[r for r in nut if r['temperature_descriptor_C']])]:
        for withT in ([False] if sample=='primary13' else [False,True]):
            for split,key in [('source','source_id'),('network','network_stress_group')]:
                scope=('noT' if sample=='primary13' else 'matched10_'+('withT' if withT else 'noT'))+'_'+split
                for hold in sorted({r[key] for r in rows}):
                    train=[r for r in rows if r[key]!=hold];test=[r for r in rows if r[key]==hold]
                    assert {r['source_id'] for r in train}.isdisjoint(r['source_id'] for r in test)
                    splits.append(dict(scope=scope,held_out=hold,training_sources=';'.join(sorted({r['source_id'] for r in train})),
                        test_sources=';'.join(sorted({r['source_id'] for r in test}))))
                    pool=m.gain_fit(train,withT,'pooled');within=m.gain_fit(train,withT,'within')
                    w0=np.array([float(r['initial_weight_g']) for r in test])
                    for model,f,lam in [('gain_pooled',pool,1),('gain_within',within,1),('zero_response_level',within,0)]:
                        final=w0+m.gain_predict(f,test,lam)
                        changed=[dict(r,final_weight_g='987654321') for r in test]
                        assert np.array_equal(final,w0+m.gain_predict(f,changed,lam))
                        predictions+=prediction_rows(test,final,scope,model,hold)
                    if scope=='noT_source':
                        rates=defaultdict(list)
                        for r in train:rates[r['source_id']].append(math.log(float(r['final_weight_g'])/float(r['initial_weight_g']))/float(r['duration_d']))
                        rate=m.mean([m.mean(v) for v in rates.values()])
                        final=[float(r['initial_weight_g'])*math.exp(rate*float(r['duration_d'])) for r in test]
                        predictions+=prediction_rows(test,final,scope,'training_study_mean_SGR',hold)
                        predictions+=prediction_rows(test,w0,scope,'no_growth',hold)
    pairs,source=evaluate(predictions);macro=summarize(source)
    for name,rows in [('gain_predictions',predictions),('gain_pairs',pairs),('gain_source_metrics',source),('gain_macro_metrics',macro),('gain_splits',splits)]:
        write(out/(name+'.csv'),rows)
    write(out/'Table_1.csv',[r for r in macro if r['scope']=='noT_source'])
    write(out/'Figure_1_source.csv',[r for r in source if r['scope']=='noT_source'])
    write(out/'Table_S5_temperature.csv',[r for r in macro if r['scope'].startswith('matched10')])
    write(out/'Table_S3_network.csv',[r for r in source if r['scope']=='noT_network'])
    ties=[]
    for model in sorted({r['model'] for r in pairs if r['scope']=='noT_source'}):
        a=[r for r in pairs if r['scope']=='noT_source' and r['model']==model]
        grouped=defaultdict(list)
        for r in a:grouped[r['source_id']].append(r)
        ties.append(dict(model=model,dependent_pairs=len(a),observed_ties=sum(r['observed_tie'] for r in a),
            predicted_ties=sum(r['predicted_tie'] for r in a),
            source_equal_predicted_tie_fraction=m.mean([m.mean([int(x['predicted_tie']) for x in g]) for g in grouped.values()])))
    write(out/'Table_S6_ties.csv',ties)
    return source,macro

def design_analysis(nut,source,out):
    spans=[]
    for sid,ids in sorted(m.bysource(nut).items()):
        cp=[float(nut[i]['diet_protein_DM_pct']) for i in ids];cl=[float(nut[i]['diet_lipid_DM_pct']) for i in ids]
        spans.append(dict(source_id=sid,CP_span=max(cp)-min(cp),CL_span=max(cl)-min(cl),maximum_span=max(max(cp)-min(cp),max(cl)-min(cl))))
    idx={(r['source_id'],r['model']):r for r in source if r['scope']=='noT_source'}
    groups=[]
    for threshold in [.5,1,2,3,5,10]:
        for label in ['gradient','other']:
            ids=[r['source_id'] for r in spans if (r['maximum_span']>threshold)==(label=='gradient')]
            for model in ['gain_pooled','gain_within','zero_response_level']:
                a=[idx[(s,model)] for s in ids]
                row=dict(threshold_pp=threshold,stratum=label,model=model,studies=len(a))
                for field in ['contrast_NMAE','direction_accuracy','response_RMS_ratio']:
                    row[field]=m.mean([float(r[field]) for r in a])
                row['studies_improved_vs_pooled']=sum(float(idx[(s,'gain_within')]['contrast_NMAE'])<float(idx[(s,'gain_pooled')]['contrast_NMAE']) for s in ids)
                groups.append(row)
    write(out/'design_spans.csv',spans);write(out/'Table_S4_thresholds.csv',groups)
    write(out/'Table_2.csv',[r for r in groups if r['threshold_pp']==2])
    continuous=[]
    for baseline in ['gain_pooled','zero_response_level']:
        x=[r['maximum_span'] for r in spans]
        y=[float(idx[(r['source_id'],baseline)]['contrast_NMAE'])-float(idx[(r['source_id'],'gain_within')]['contrast_NMAE']) for r in spans]
        continuous.append(dict(baseline=baseline,spearman_rho=float(spearmanr(x,y).statistic)))
    write(out/'continuous_span.csv',continuous)

def sgr_fit_predict(train,test,kind):
    def xmat(rows):
        return np.array([[math.log(float(r['initial_weight_g'])),math.log(float(r['duration_d']))]+
            ([] if kind=='context_SGR' else [float(r['diet_protein_DM_pct']),float(r['diet_lipid_DM_pct'])]) for r in rows])
    w=m.weights(train);x=xmat(train);q=xmat(test);mu=w@x;sd=np.sqrt(w@((x-mu)**2));active=sd>=1e-12;sd[~active]=1
    z=(x-mu)/sd;z[:,~active]=0;qz=(q-mu)/sd;qz[:,~active]=0
    if kind=='quadratic_nutrition_SGR':
        squares=z[:,-2:]**2;qs=qz[:,-2:]**2;sm=w@squares;ss=np.sqrt(w@((squares-sm)**2));a=ss>=1e-12;ss[~a]=1
        zs=(squares-sm)/ss;zs[:,~a]=0;zz=(qs-sm)/ss;zz[:,~a]=0;z=np.c_[z,zs];qz=np.c_[qz,zz]
    y=np.array([math.log(float(r['final_weight_g'])/float(r['initial_weight_g']))/float(r['duration_d']) for r in train])
    ym=float(w@y);A=z.T@(w[:,None]*z)+np.eye(z.shape[1]);b=z.T@(w*(y-ym));beta=np.linalg.solve(A,b);yp=ym+qz@beta
    return [float(r['initial_weight_g'])*math.exp(float(g)*float(r['duration_d'])) for r,g in zip(test,yp)]

def reference_analysis(nut,studies,out):
    designated={r['source_id']:r['terminal_reference_treatment'] for r in studies if r['terminal_reference_treatment']}
    pred=[];rotations=[];summary=[]
    for sid,ids in sorted(m.bysource(nut).items()):
        train=[r for r in nut if r['source_id']!=sid];test=[nut[i] for i in ids]
        for model in ['context_SGR','nutrition_SGR','quadratic_nutrition_SGR']:
            final=sgr_fit_predict(train,test,model)
            for j,ref in enumerate(test):
                offset=float(ref['final_weight_g'])-final[j]
                others=[(r,p) for i,(r,p) in enumerate(zip(test,final)) if i!=j]
                mae0=m.mean([abs(p-float(r['final_weight_g'])) for r,p in others])
                mae=m.mean([abs(p+offset-float(r['final_weight_g'])) for r,p in others])
                before=m.mean([abs(p-float(r['final_weight_g']))/float(r['final_weight_g']) for r,p in others])
                after=m.mean([abs(p+offset-float(r['final_weight_g']))/float(r['final_weight_g']) for r,p in others])
                selected=ref['treatment_id']==designated[sid]
                rotations.append(dict(source_id=sid,model=model,reference_treatment=ref['treatment_id'],designated=selected,
                    endpoint_MAPE_before=before,endpoint_MAPE_after=after,endpoint_MAE_g_before=mae0,endpoint_MAE_g_after=mae))
                if selected:
                    for r,p in others:
                        pred.append(dict(source_id=sid,treatment_id=r['treatment_id'],model=model,
                            reference_treatment=ref['treatment_id'],observed_final_weight_g=r['final_weight_g'],
                            predicted_before_g=p,predicted_after_g=p+offset))
    for model in ['context_SGR','nutrition_SGR','quadratic_nutrition_SGR']:
        a=[r for r in rotations if r['model']==model and r['designated']]
        summary.append(dict(model=model,studies=len(a),
            **{k:m.mean([r[k] for r in a]) for k in ['endpoint_MAPE_before','endpoint_MAPE_after','endpoint_MAE_g_before','endpoint_MAE_g_after']}))
    write(out/'Figure_3_summary.csv',summary);write(out/'reference_rotations.csv',rotations);write(out/'reference_predictions.csv',pred)
    write(out/'Table_S7_references.csv',[dict(source_id=r['source_id'],reference_treatment=r['terminal_reference_treatment'],rule=r['terminal_reference_rule']) for r in studies if r['terminal_reference_treatment']])

def interval_predictor(task,train):
    if task=='nutrition_gain_pooled':
        f=m.gain_fit(train,False)
        return lambda rows:np.array([float(r['initial_weight_g']) for r in rows])+m.gain_predict(f,rows)
    f=m.core_fit(train)
    return lambda rows:m.core_predict(f,rows)

def calibration_analysis(core,nut,out):
    results=[];endpoints=[];splits=[]
    for task,rows in [('nutrition_gain_pooled',nut),('core_size_SGR',core)]:
        for split,key in [('source','source_id'),('network','network_stress_group')]:
            scope=('noT_' if task.startswith('nutrition') else 'core_')+split
            units=sorted({r[key] for r in rows})
            for hold in units:
                test=[r for r in rows if r[key]==hold];available=[u for u in units if u!=hold]
                train_full=[r for r in rows if r[key]!=hold]
                configs=[]
                predict=interval_predictor(task,train_full);test_pred=predict(test);scores=[]
                for omitted in available:
                    inner=[r for r in rows if r[key] not in {hold,omitted}];cal=[r for r in rows if r[key]==omitted]
                    p=interval_predictor(task,inner)(cal);means=m.msource(cal)
                    scores.append(max(abs(float(y)-float(r['final_weight_g']))/means[r['source_id']] for r,y in zip(cal,p)))
                configs.append(('cross', '',test_pred,sorted(scores),len(available)))
                order=sorted(available,key=lambda u:hashlib.sha256(('frontier-split-v1|'+u).encode()).hexdigest())
                for frac in [.5,.75]:
                    nc=min(len(order)-1,math.ceil(len(order)*frac));cu=set(order[:nc]);tu=set(order[nc:])
                    train=[r for r in rows if r[key] in tu];cal=[r for r in rows if r[key] in cu]
                    predict=interval_predictor(task,train);cp=predict(cal);tp=predict(test);means=m.msource(cal);scores=defaultdict(list)
                    for r,p in zip(cal,cp):scores[r[key]].append(abs(float(p)-float(r['final_weight_g']))/means[r['source_id']])
                    configs.append(('split',frac,tp,sorted(max(a) for a in scores.values()),len(tu)))
                    splits.append(dict(task=task,scope=scope,held_out=hold,calibration_fraction=frac,
                        training_units=';'.join(sorted(tu)),calibration_units=';'.join(sorted(cu))))
                means=m.msource(test)
                for method,frac,tp,scores,nfit in configs:
                    nc=len(scores)
                    for coverage in [.8,.9,.95]:
                        rank=math.ceil((nc+1)*coverage-1e-12);q=scores[rank-1] if rank<=nc else math.inf
                        bundle=[]
                        for sid,ids in sorted(m.bysource(test).items()):
                            observed=[float(test[i]['final_weight_g']) for i in ids];predicted=[float(tp[i]) for i in ids];initial=means[sid]
                            maxerr=max(abs(p-y)/initial for p,y in zip(predicted,observed));bundle.append(maxerr<=q)
                            mean_pred=m.mean([abs(p) for p in predicted]);relative=q*initial/mean_pred if mean_pred else math.inf
                            results.append(dict(method=method,task=task,scope=scope,held_out=hold,calibration_fraction=frac,
                                nominal_coverage=coverage,source_id=sid,endpoints=len(ids),quantile_rank=rank,
                                calibration_units=nc,fitting_units=nfit,quantile=q,finite_interval=math.isfinite(q),
                                source_all_endpoints_covered=maxerr<=q,source_max_normalized_error=maxerr,
                                endpoint_MAPE=m.mean([abs(p-y)/y for p,y in zip(predicted,observed)]),
                                normalized_interval_width=2*q,terminal_relative_halfwidth=relative,terminal_accepted=relative<=.25))
                            if method=='cross' and coverage==.8 and split=='source' and task=='core_size_SGR':
                                for i,p,y in zip(ids,predicted,observed):
                                    endpoints.append(dict(source_id=sid,treatment_id=test[i]['treatment_id'],
                                        observed_final_weight_g=y,predicted_final_weight_g=p,lower_g=p-q*initial,
                                        upper_g=p+q*initial,covered=abs(p-y)<=q*initial))
                        for r in results[-len(means):]:r['test_bundle_covered']=all(bundle)
    write(out/'calibration_results.csv',results);write(out/'calibration_splits.csv',splits)
    write(out/'Figure_4b_endpoints.csv',endpoints)
    summary=[];groups=defaultdict(list)
    for r in results:groups[(r['method'],r['task'],r['scope'],r['calibration_fraction'],r['nominal_coverage'])].append(r)
    for key,a in groups.items():
        bundles={r['held_out']:r['test_bundle_covered'] for r in a}
        summary.append(dict(zip(['method','task','scope','calibration_fraction','nominal_coverage'],key),
            evaluated_sources=len(a),held_out_bundles=len(bundles),finite_bundles=len({r['held_out'] for r in a if r['finite_interval']}),
            covered_sources=sum(r['source_all_endpoints_covered'] for r in a),covered_bundles=sum(bundles.values()),
            endpoint_MAPE=m.mean([r['endpoint_MAPE'] for r in a]),normalized_interval_width=m.mean([r['normalized_interval_width'] for r in a]),
            accepted_sources=sum(r['terminal_accepted'] for r in a)))
    write(out/'calibration_summary.csv',summary)
    write(out/'Table_3.csv',[r for r in summary if r['scope'] in ['noT_source','core_source'] and r['nominal_coverage']==.9 and
        (r['method']=='cross' or r['calibration_fraction']==(.75 if r['task'].startswith('nutrition') else .5))])

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('outputs'));a=p.parse_args()
    data=a.data_dir.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    paths=[data/n for n in ['core.csv','nutrition.csv','studies.csv','provenance.csv']]
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    core,nut,studies,provenance=[read(f) for f in paths]
    assert len({r['source_id'] for r in core})==18 and len(core)==91
    assert len({r['source_id'] for r in nut})==13 and len(nut)==69 and len(studies)==22
    def keys(rows):return {(r['source_id'],r['treatment_id'],float(r['duration_d'])) for r in rows}
    ca,na=keys(core),keys(nut)
    write(out/'Table_S1_datasets.csv',[dict(relation=name,studies=len(sources),endpoints=len(records)) for name,sources,records in [
        ('core',{r['source_id'] for r in core},ca),('nutrition',{r['source_id'] for r in nut},na),
        ('intersection',{s for s,t,d in ca&na},ca&na),('union',{s for s,t,d in ca|na},ca|na)]])
    write(out/'Table_S2_and_S8_studies.csv',studies)
    print('Refitting pooled, within-study and baseline models...',flush=True)
    source,macro=gain_analysis(nut,out);design_analysis(nut,source,out)
    print('Refitting reference models and REML summaries...',flush=True)
    reference_analysis(nut,studies,out);write(out/'heterogeneity_ICC.csv',heterogeneity(nut))
    print('Refitting split and cross-calibration models...',flush=True)
    calibration_analysis(core,nut,out)
    assert all(hashlib.sha256(f.read_bytes()).hexdigest()==hashes[f.name] for f in paths)
    (out/'run_summary.json').write_text(json.dumps(dict(input_sha256=hashes,input_files_unchanged=True,
        source_holdout_and_fold_local_transforms=True,all_models_refitted=True,
        frozen_prediction_files_read=False,empirical_intervals_without_coverage_guarantee=True),indent=2)+'\n',encoding='utf-8')
    print('Completed: '+str(out),flush=True)

if __name__=='__main__':main()

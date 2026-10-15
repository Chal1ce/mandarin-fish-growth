"""Study-equal evaluation; treatment pairs are dependent."""
from collections import defaultdict
from itertools import combinations
import math
from models import mean

def evaluate(pred):
    groups=defaultdict(list);pairs=[];source=[]
    for r in pred:groups[(r['scope'],r['model'],r['source_id'])].append(r)
    for (scope,model,sid),aa in sorted(groups.items()):
        strata=defaultdict(list)
        for r in aa:strata[float(r['duration_d'])].append(r)
        metrics=[]
        for days,a in sorted(strata.items()):
            a=sorted(a,key=lambda r:r['treatment_id']);m=mean([float(r['initial_weight_g']) for r in a]);pp=[]
            for x,y in combinations(a,2):
                ob=float(x['reported_gain_g'])-float(y['reported_gain_g']);pr=float(x['predicted_gain_g'])-float(y['predicted_gain_g'])
                sign=lambda v:0 if abs(v)<=1e-8 else (1 if v>0 else -1)
                row=dict(scope=scope,model=model,source_id=sid,duration_d=days,treatment_i=x['treatment_id'],treatment_j=y['treatment_id'],
                    observed_gain_difference_g=ob,predicted_gain_difference_g=pr,error_over_mean_initial=abs(pr-ob)/m,
                    absolute_error_g=abs(pr-ob),direction_correct=sign(ob)==sign(pr),observed_tie=sign(ob)==0,predicted_tie=sign(pr)==0,)
                pp.append(row);pairs.append(row)
            if pp:
                nonzero=[r for r in pp if not r['observed_tie']]
                obsr=math.sqrt(mean([r['observed_gain_difference_g']**2 for r in pp]));prr=math.sqrt(mean([r['predicted_gain_difference_g']**2 for r in pp]))
                best=max(a,key=lambda r:float(r['predicted_gain_g'])) # sorted IDs give fixed tie order
                regret=(max(float(r['reported_gain_g']) for r in a)-float(best['reported_gain_g']))/m
                metrics.append(dict(contrast_NMAE=mean([r['error_over_mean_initial'] for r in pp]),
                    contrast_MAE_g=mean([r['absolute_error_g'] for r in pp]),
                    direction_accuracy=mean([int(r['direction_correct']) for r in nonzero]) if nonzero else '',
                    response_RMS_ratio=prr/obsr if obsr else '',choice_regret_over_initial=regret,
                    choice_reported_best=int(regret<=1e-10),chosen_treatment=best['treatment_id']))
        row=dict(scope=scope,model=model,source_id=sid,network_stress_group=aa[0]['network_stress_group'],endpoints=len(aa),
            endpoint_MAPE=mean([abs(float(r['predicted_final_weight_g'])-float(r['reported_final_weight_g']))/float(r['reported_final_weight_g']) for r in aa]),
            endpoint_MAE_g=mean([abs(float(r['predicted_final_weight_g'])-float(r['reported_final_weight_g'])) for r in aa]),
            negative_final_predictions=sum(float(r['predicted_final_weight_g'])<0 for r in aa),
            predicted_weight_loss_positions=sum(float(r['predicted_gain_g'])<0 for r in aa))
        for k in ['contrast_NMAE','contrast_MAE_g','direction_accuracy','response_RMS_ratio','choice_regret_over_initial','choice_reported_best']:
            vals=[float(r[k]) for r in metrics if r[k]!=''];row[k]=mean(vals) if vals else ''
        row['chosen_treatments']=';'.join(r['chosen_treatment'] for r in metrics);source.append(row)
    return pairs,source

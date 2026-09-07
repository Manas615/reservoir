import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
def evaluate_outputs(y_true,y_prob,zone_true=None,zone_pred=None,uncertainty=None):
 y=np.asarray(y_true).ravel(); p=np.asarray(y_prob).ravel(); b=p>=.5; out={"iou":float(np.logical_and(y>.5,b).sum()/max(np.logical_or(y>.5,b).sum(),1)),"dice":float(2*np.logical_and(y>.5,b).sum()/max((y>.5).sum()+b.sum(),1)),"precision":float(precision_score(y>.5,b,zero_division=0)),"recall":float(recall_score(y>.5,b,zero_division=0)),"f1":float(f1_score(y>.5,b,zero_division=0))}
 if len(np.unique(y))>1: out["auc"]=float(roc_auc_score(y,p))
 if zone_true is not None: out.update(mae=float(mean_absolute_error(zone_true,zone_pred)),rmse=float(mean_squared_error(zone_true,zone_pred)**.5),r2=float(r2_score(zone_true,zone_pred)))
 if uncertainty is not None: out["uncertainty_error_correlation"]=float(np.corrcoef(np.abs(y-p),np.asarray(uncertainty).ravel())[0,1])
 return out

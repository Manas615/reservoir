import numpy as np
def adaptive_hotspot_threshold(turbidity,rainfall=0.,flow=0.,season=0.,historical_turbidity=None,reservoir_condition=.5):
 base=np.quantile(np.asarray(historical_turbidity if historical_turbidity is not None else turbidity),.85); factor=1+.10*np.clip(rainfall,0,1)+.08*np.clip(flow,0,1)+.04*np.sin(2*np.pi*season)+.04*(reservoir_condition-.5); return float(np.clip(base*factor,0,1))
def generate_derived_labels(turbidity,**context):
 x=np.asarray(turbidity); threshold=adaptive_hotspot_threshold(x,**context); return {"fixed_label":(x>=np.quantile(x,.85)).astype("uint8"),"adaptive_label":(x>=threshold).astype("uint8"),"threshold":threshold,"label_type":"derived_satellite_observation"}

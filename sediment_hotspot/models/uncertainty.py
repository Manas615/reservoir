import torch
def mc_dropout_predict(model,batch,samples=20):
 model.eval(); [m.train() for m in model.modules() if m.__class__.__name__.startswith("Dropout")]; p=torch.stack([model(batch)["heatmap"] for _ in range(samples)]); return p.mean(0),p.var(0,unbiased=False)

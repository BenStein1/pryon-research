"""NumPy head-only QAT: exact Pryon quantized forward, straight-through gradients.

Recording labels supervise aged three-row probability windows. Encoder frozen.
No held-out data, microphone, enrollment state or model file mutation here.
"""
import numpy as np
from scipy.special import expit, logsumexp

from pryon_head_quantize import quantize_row
from pryon_quant_model import quantize_activation_rows
from train_pryon_streaming import temporal_windows


def prepare_rows(x):
    x=np.asarray(x,np.float32)
    shape=x.shape[:-1]
    if x.shape[-1]!=100 or not np.isfinite(x).all():
        raise ValueError('Expected finite 100-D activations')
    flat=x.reshape(-1,100)
    q,s,const=quantize_activation_rows(flat)
    return dict(x=flat,q=q.astype(np.float64),s=s,const=const,shape=shape)


def quantized_logits(rows, weight, bias):
    """Exact one-class quant_matmul epilogue, using cached activation quantization."""
    q,sw,zw,_,_=quantize_row(np.asarray(weight,np.float32))
    wd=q.astype(np.float64)-zw
    acc=rows['q'] @ wd
    y=acc.astype(np.float32)*(rows['s']*sw)
    const=rows['const']
    if const.any():
        y[const]=rows['x'][const].astype(np.float64) @ (wd*sw)
    return (y+np.float32(bias)).reshape(rows['shape'])


def window_loss(logits, baseline, label, temperature=.02, pool="smooth"):
    """Recording BCE after width-3 probability mean and smooth or exact maximum.

    Returns a logit gradient (selected-window subgradient for max) for STE;
    this does not differentiate the integer quantizer itself.
    """
    p=expit(logits.astype(np.float64)-baseline)
    scores=p.mean(-1)
    if pool == 'max':
        alpha=np.zeros_like(scores)
        alpha.flat[int(np.argmax(scores))]=1.
        score=np.max(scores)
    elif pool == 'smooth':
        z=scores/temperature
        lse=logsumexp(z)
        alpha=np.exp(z-lse)
        score=temperature*(lse-np.log(scores.size))
    else:
        raise ValueError('Unknown temporal pooling rule')
    score=np.clip(score,1e-12,1-1e-12)
    loss=-label*np.log(score)-(1-label)*np.log1p(-score)
    derivative=(score-label)/(score*(1-score))
    dp=derivative*alpha[...,None]*p*(1-p)/3
    return float(loss),dp


def train_qat(activations, labels, fixed_weights, fixed_bias, initial_weight,
              initial_bias, steps=800, learning_rate=.01, regularization=1e-5,negative_pool="smooth",
              batch_size=None,class_balance=False,eval_interval=50,seed=0):
    """Optimize float shadow parameters; evaluate exported int8 weights each step.

    Adam with STE (ignore derivatives through quantizer scale/rounding), best
    training-loss checkpoint. No assertion of smooth-optimizer convergence.
    Final float shadow row is exported by the existing quantize_row routine.
    """
    if negative_pool not in ('smooth','max'):raise ValueError('Unknown negative pooling rule')
    labels=np.asarray(labels,np.float64)
    if (len(activations)!=len(labels) or set(labels)!={0.,1.}
            or steps<1 or not np.isfinite(learning_rate) or learning_rate<=0
            or not np.isfinite(regularization) or regularization<=0):
        raise ValueError('Require both labels, positive steps/rate/regularization')
    if batch_size is not None and (batch_size<2 or batch_size%2 or eval_interval<1):
        raise ValueError('Require an even batch size >=2 and positive evaluation interval')
    rng=np.random.default_rng(seed)
    classes=[np.flatnonzero(labels==value) for value in (0,1)]
    windows=[temporal_windows(a) for a in activations]
    if any(not len(x) for x in windows):raise ValueError('Insufficient post-warmup cycles')
    prepared=[prepare_rows(x) for x in windows]
    flat=np.concatenate([x.reshape(-1,100) for x in windows])
    mean=flat.mean(0).astype(np.float64)
    std=np.maximum(flat.std(0),1e-3).astype(np.float64)
    # Fixed competitor rows use the exact inference kernel, including their
    # original per-row quantization metadata, supplied by the caller below.
    from pryon_quant_model import quant_matmul
    wq,sw,zw=fixed_weights
    baseline=[]
    gradient_rows=[]
    for row in prepared:
        competitors=quant_matmul(row['x'],wq,sw,zw,{'constant_rows':0})+fixed_bias
        baseline.append(np.logaddexp(competitors[:,0].astype(np.float64),
                                     competitors[:,1].astype(np.float64)).reshape(row['shape']))
        qdq=row['q']*row['s'][:,None]
        qdq[row['const']]=row['x'][row['const']]
        gradient_rows.append((qdq-mean)/std)
    theta=np.r_[np.asarray(initial_weight)*std,initial_bias+mean@initial_weight]
    prior=theta.copy()
    first_moment=np.zeros_like(theta);second_moment=np.zeros_like(theta)
    best=None;trace=[]
    def evaluate(t, need_gradient=True,indices=None):
        w=t[:-1]/std;b=t[-1]-mean@w
        loss=0.;grad=np.zeros_like(t)
        chosen=range(len(labels)) if indices is None else indices
        for i in chosen:
            row,base,x,label=prepared[i],baseline[i],gradient_rows[i],labels[i]
            factor=(len(labels)/(2*len(classes[int(label)]))) if class_balance and indices is None else 1.
            value,dp=window_loss(quantized_logits(row,w,b),base,label,pool=negative_pool if label==0 else "smooth")
            loss+=factor*value/len(chosen)
            if need_gradient:
                d=dp.reshape(-1)*factor/len(chosen)
                grad[:-1]+=d@x
                grad[-1]+=d.sum()
        delta=t[:-1]-prior[:-1]
        loss+=.5*regularization*(delta@delta)
        grad[:-1]+=regularization*delta
        return float(loss),grad
    for step in range(steps+1):
        checkpoint=batch_size is None or step%eval_interval==0 or step==steps
        if batch_size is None:
            loss,grad=evaluate(theta)
        else:
            if checkpoint:loss,_=evaluate(theta,False)
            indices=(np.concatenate([rng.choice(group,batch_size//2,replace=True) for group in classes])
                     if class_balance else rng.choice(len(labels),batch_size,replace=True))
            _,grad=evaluate(theta,True,indices)
        if not np.isfinite(grad).all() or (checkpoint and not np.isfinite(loss)):
            raise RuntimeError('Nonfinite QAT update')
        if checkpoint and (best is None or loss<best[0]):best=(loss,theta.copy(),step)
        if checkpoint and (step%100==0 or step==steps):trace.append(dict(step=step,loss=loss,best_loss=best[0]))
        if step==steps:break
        # Bound catastrophic surrogate updates; integer forward still decides
        # which checkpoint wins. Gradient clipping is not an acceptance test.
        grad*=min(1.,10./max(float(np.linalg.norm(grad)),1e-12))
        first_moment=.9*first_moment+.1*grad
        second_moment=.999*second_moment+.001*grad*grad
        theta-=learning_rate*(first_moment/(1-.9**(step+1)))/(np.sqrt(second_moment/(1-.999**(step+1)))+1e-8)
    w=best[1][:-1]/std;b=best[1][-1]-mean@w
    # Cast exactly as the export API does, and record the resulting forward loss.
    w=w.astype(np.float32);b=np.float32(b)
    exported=np.r_[w.astype(np.float64)*std,float(b)+mean@w]
    export_loss,_=evaluate(exported,False)
    return w,b,dict(method='exact int8 forward / STE / Adam',steps=steps,
        learning_rate=learning_rate,regularization=regularization,negative_pool=negative_pool,
        batch_size=batch_size,class_balance=class_balance,eval_interval=eval_interval,seed=seed,
        initial_loss=trace[0]['loss'],best_shadow_loss=best[0],best_step=best[2],
        exported_loss=export_loss,trace=trace,
        selection='training loss only; no validation access',
        limitations='STE is approximate; full encoder frozen; final validation must use independent engine')

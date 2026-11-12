"""Experimental QAT of fc_ww_1 bias + wake output row; other tensors frozen."""
import copy
import numpy as np
from onnx import numpy_helper

from pryon_head_quantize import quantize_row,replace_original_pryon_wake_row
from pryon_quant_model import quant_matmul,quantize_activation_rows,P
from pryon_qat import window_loss
from train_pryon_streaming import temporal_windows

DENSE_BIAS=P+'fc_ww_1/BiasAdd/ReadVariableOp:0'


def logits_loss_gradient(logits,label):
    """All three logits matter when upstream activations change."""
    baseline=np.logaddexp(logits[...,0].astype(np.float64),logits[...,2].astype(np.float64))
    loss,d=window_loss(logits[...,1],baseline,label)
    g=np.zeros_like(logits,dtype=np.float64)
    g[...,1]=d
    g[...,0]=-d*np.exp(logits[...,0]-baseline)
    g[...,2]=-d*np.exp(logits[...,2]-baseline)
    return loss,g


def replace_dense_bias_and_head(source,dense_bias,weight,bias):
    dense_bias=np.asarray(dense_bias,np.float32)
    if dense_bias.shape!=(100,) or not np.isfinite(dense_bias).all():
        raise ValueError('Expected finite 100-D dense bias')
    candidate,info=replace_original_pryon_wake_row(source,weight,bias)
    tensor=next(t for t in candidate.graph.initializer if t.name==DENSE_BIAS)
    if tuple(tensor.dims)!=(100,):raise ValueError('Dense bias shape changed')
    tensor.CopyFrom(numpy_helper.from_array(dense_bias,name=DENSE_BIAS))
    check_dense_bias_and_head(source,candidate)
    return candidate,info


def check_dense_bias_and_head(source,candidate):
    from pryon_training_io import check_original_head_only
    old=next(t for t in source.graph.initializer if t.name==DENSE_BIAS)
    restored=copy.deepcopy(candidate)
    new=next(t for t in restored.graph.initializer if t.name==DENSE_BIAS)
    a=numpy_helper.to_array(new)
    if a.dtype!=np.float32 or a.shape!=(100,) or not np.isfinite(a).all():
        raise ValueError('Invalid adapted dense bias')
    new.CopyFrom(old)
    # Existing byte-level guard then proves every other graph/weight/state
    # byte unchanged except the five tensors' target-class entries.
    return check_original_head_only(source,restored)


def train_dense(bases,labels,head,dense_bias,initial_weight,initial_bias,
                     steps=800,learning_rate=.01,head_reg=1e-5,dense_reg=1e-3,
                hidden_inputs=None,dense_matrix=None,matrix_reg=1e-3):
    """bases: frozen fc_ww_1 quantized GEMM outputs BEFORE its separate bias.

    All forward kernels are the independent quant_matmul implementation.
    STE propagates through ReLU/input quantization and all three output logits.
    Adam/checkpoint selection sees TRAIN recordings only, not validation.
    """
    if len(bases)!=len(labels) or set(labels)!={0,1}:raise ValueError('Need both labels')
    windows=[temporal_windows(a) for a in bases]
    if any(x.shape[-1]!=100 or not len(x) or not np.isfinite(x).all() for x in windows):
        raise ValueError('Invalid dense preactivation windows')
    db=np.asarray(dense_bias,np.float32)
    initial=np.concatenate([np.maximum(x+db,0).reshape(-1,100) for x in windows]).astype(np.float64)
    mean=initial.mean(0);std=np.maximum(initial.std(0),1e-3)
    hidden_std=np.maximum(np.concatenate([x.reshape(-1,100) for x in windows]).std(0),.1).astype(np.float64)
    prior=np.r_[np.asarray(initial_weight)*std,initial_bias+mean@initial_weight,np.zeros(100)]
    matrix_mode=dense_matrix is not None
    hidden=[]
    if matrix_mode:
        if hidden_inputs is None:raise ValueError('Matrix adaptation requires frozen hidden inputs')
        mq,ms,mz=dense_matrix
        if mq.shape!=(100,100) or mq.dtype!=np.int8:raise ValueError('Expected 100x100 int8 dense matrix')
        for x in hidden_inputs:
            hx=temporal_windows(x).reshape(-1,100)
            hq,hs,hconst=quantize_activation_rows(hx)
            hqdq=hq.astype(np.float64)*hs[:,None];hqdq[hconst]=hx[hconst]
            hidden.append((hx,hq.astype(np.float64),hs,hconst,hqdq))
        prior=np.r_[prior,np.zeros(10000)]
    theta=prior.copy();m=np.zeros_like(theta);v=np.zeros_like(theta);best=None;trace=[]
    def evaluate(t):
        w=(t[:100]/std).astype(np.float32)
        b=np.float32(t[100]-mean@(t[:100]/std))
        bh=(db+t[101:201]*hidden_std).astype(np.float32)
        matrix_q=None
        if matrix_mode:
            matrix_q=np.clip(np.rint(mq.astype(np.float64)+t[201:].reshape(100,100)),-128,127).astype(np.int8)
        q,sw,zw,bias=(a.copy() for a in head)
        q[1],sw[1],zw[1],_,_=quantize_row(w);bias[1]=b
        wd=(q.astype(np.float64)-zw[:,None])*sw[:,None]
        loss=0.;grad=np.zeros_like(t)
        for bag,(base,label) in enumerate(zip(windows,labels)):
            if matrix_mode:
                hx,hq,hs,hconst,hqdq=hidden[bag]
                md=matrix_q.astype(np.float64)-mz[:,None]
                y=(hq@md.T).astype(np.float32)*(hs[:,None]*ms)
                if hconst.any():y[hconst]=hx[hconst].astype(np.float64)@(md*ms[:,None]).T
                base=y.reshape(base.shape)
            z=base+bh;a=np.maximum(z,0)
            logits=(quant_matmul(a.reshape(-1,100),q,sw,zw,{'constant_rows':0})+bias).reshape(*a.shape[:-1],3)
            value,gl=logits_loss_gradient(logits,label)
            gl=gl.reshape(-1,3)/len(labels);loss+=value/len(labels)
            qa,scale,const=quantize_activation_rows(a.reshape(-1,100))
            aq=qa.astype(np.float64)*scale[:,None]
            aq[const]=a.reshape(-1,100)[const]
            grad[:100]+=gl[:,1]@((aq-mean)/std)
            grad[100]+=gl[:,1].sum()
            dz=(gl@wd)*(z.reshape(-1,100)>0)
            grad[101:201]+=dz.sum(0)*hidden_std
            if matrix_mode:
                grad[201:]+=((dz.T@hqdq)*ms[:,None]).ravel()
        delta=t[:101]-prior[:101]
        loss+=.5*head_reg*(delta@delta)+.5*dense_reg*(t[101:201]@t[101:201])
        grad[:101]+=head_reg*delta;grad[101:201]+=dense_reg*t[101:201]
        if matrix_mode:
            loss+=.5*matrix_reg*np.mean(t[201:]**2)
            grad[201:]+=matrix_reg*t[201:]/10000
        return float(loss),grad,(bh,w,b,matrix_q)
    for step in range(steps+1):
        loss,g,params=evaluate(theta)
        if not np.isfinite(loss) or not np.isfinite(g).all():raise RuntimeError('Nonfinite dense-adaptation step')
        if best is None or loss<best[0]:best=(loss,theta.copy(),step,params)
        if step%100==0 or step==steps:trace.append(dict(step=step,loss=loss,best_loss=best[0]))
        if step==steps:break
        g*=min(1.,10/max(float(np.linalg.norm(g)),1e-12))
        m=.9*m+.1*g;v=.999*v+.001*g*g
        theta-=learning_rate*(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.999**(step+1)))+1e-8)
        if matrix_mode:
            theta[201:]=np.clip(theta[201:],(-128-mq.astype(np.float64)).ravel(),(127-mq.astype(np.float64)).ravel())
    bh,w,b,matrix_q=best[3]
    return bh,w,b,matrix_q,dict(method='fc_ww_1 '+('matrix+bias' if matrix_mode else 'bias')+' + wake row QAT; exact forward, STE/Adam',
        steps=steps,learning_rate=learning_rate,head_reg=head_reg,dense_reg=dense_reg,
        best_step=best[2],initial_loss=trace[0]['loss'],best_loss=best[0],trace=trace,
        dense_bias_max_delta=float(np.max(abs(bh-db))),
        matrix_reg=matrix_reg if matrix_mode else None,
        changed_dense_weight_codes=int(np.count_nonzero(matrix_q!=mq)) if matrix_mode else 0,
        trainable_parameters=len(theta),selection='training loss only',
        frozen='Conv/LSTM/SE and dense quantization scales/zero-points unchanged')


def dense_tensor_names(source):
    nodes=[n for n in source.graph.node if n.op_type=='HybridGemm' and '/fc_ww_1/' in n.name]
    if len(nodes)!=1 or len(nodes[0].input)!=5:raise ValueError('Unexpected fc_ww_1 topology')
    return list(nodes[0].input[1:])


def check_dense_layer_and_head(source,candidate):
    from pryon_training_io import check_original_head_only
    names=dense_tensor_names(source)
    old={t.name:t for t in source.graph.initializer}
    restored=copy.deepcopy(candidate)
    new={t.name:t for t in restored.graph.initializer}
    q=numpy_helper.to_array(new[names[0]])
    sums=numpy_helper.to_array(new[names[3]])
    bias=numpy_helper.to_array(new[DENSE_BIAS])
    if q.dtype!=np.int8 or q.shape!=(100,100):raise ValueError('Invalid dense matrix')
    if sums.dtype!=np.int32 or not np.array_equal(sums,q.astype(np.int32).sum(1)):
        raise ValueError('Dense weight sums do not match')
    if bias.shape!=(100,) or bias.dtype!=np.float32 or not np.isfinite(bias).all():
        raise ValueError('Invalid dense bias')
    for name in (names[0],names[3],DENSE_BIAS):new[name].CopyFrom(old[name])
    # Scale/zp, nodes, all other tensors and non-target output classes must still
    # be byte-identical; only matrix codes, matching sums and bias are allowed.
    return check_original_head_only(source,restored)


def replace_dense_layer_and_head(source,matrix,dense_bias,weight,bias):
    matrix=np.asarray(matrix)
    if matrix.shape!=(100,100) or matrix.dtype!=np.int8:raise ValueError('Expected int8 dense matrix')
    candidate,_=replace_original_pryon_wake_row(source,weight,bias)
    names=dense_tensor_names(source)
    updates={names[0]:matrix,names[3]:matrix.astype(np.int32).sum(1).astype(np.int32),
             DENSE_BIAS:np.asarray(dense_bias,np.float32)}
    for tensor in candidate.graph.initializer:
        if tensor.name in updates:tensor.CopyFrom(numpy_helper.from_array(updates[tensor.name],name=tensor.name))
    check_dense_layer_and_head(source,candidate)
    return candidate

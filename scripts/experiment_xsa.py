"""Avoid repeated normalization of identical value heads in XSA."""
import json
from pathlib import Path
import mlx.core as mx
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.fused import xsa
from dwarf_violetto.bench import measure
from dwarf_violetto.sampling import make_sampler
from dwarf_violetto.provenance import provenance


@mx.compile
def grouped_xsa(y, v, alpha):
    b, length, heads, dim = y.shape
    kv_heads = v.shape[-2]
    vf = v.astype(mx.float32)
    norm = mx.maximum(mx.sqrt(mx.sum(vf*vf, axis=-1, keepdims=True)), 1e-4)
    vn = (vf/norm)[:, :, :, None, :]
    yg = y.reshape(b, length, kv_heads, heads//kv_heads, dim)
    proj = mx.sum(yg.astype(mx.float32)*vn, axis=-1, keepdims=True)
    a = mx.tanh(alpha.astype(mx.float32)).reshape(1, 1, kv_heads, heads//kv_heads, 1)
    return (yg-(a*proj*vn).astype(y.dtype)).reshape(y.shape)


def main():
    model, tokenizer, _ = load('models/q8', optimized=True, fused=True)
    def enable(value):
        for layer in model.layers:
            layer.self_attn._fused_xsa = grouped_xsa if value else xsa
    prompt = prompt_tokens(tokenizer, 'Find the last three digits of 1! + 2! + ... + 20!.')
    result = {'sha256':provenance('models/q8'), 'checks':[], 'runs':[]}
    for length in [1,128,1100,8192]:
        ids=(prompt*(length//len(prompt)+1))[:length]
        caches=[model.make_cache(),model.make_cache()]
        for offset in range(0,length,512):
            for mode,cache in zip([False,True],caches):
                enable(mode); logits=model(mx.array([ids[offset:offset+512]]),cache);mx.eval(logits)
            mx.clear_cache()
        equal=True; max_abs=0
        for token in (prompt*4)[:128]:
            enable(False);a=model(mx.array([[token]]),caches[0]);mx.eval(a)
            enable(True);b=model(mx.array([[token]]),caches[1]);mx.eval(b)
            max_abs=max(max_abs,float(mx.max(mx.abs(a-b))))
            equal=equal and bool(mx.array_equal(a,b))
        result['checks'].append({'prefix':length,'positions':128,'exact':equal,'max_abs':max_abs})
        print(result['checks'][-1],flush=True)
        Path('results/decode-xsa-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
        if not equal:
            raise AssertionError(result['checks'][-1])
        del caches, logits, a, b
        mx.clear_cache()
    ids=(prompt*4)[:128];sampler=make_sampler(.6,top_p=.95)
    for mode in [False,True,True,False,False,True]:
        enable(mode);measure(model,ids,32,512,sampler)
        row=measure(model,ids,256,512,sampler);row['mode']='grouped_xsa' if mode else 'baseline'
        result['runs'].append(row)
        Path('results/decode-xsa-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(row['mode'],round(row['decode_tps'],2),flush=True)
    result['sampled_trace_identical']=all(r['tokens']==result['runs'][0]['tokens'] for r in result['runs'])
    Path('results/decode-xsa-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Sampled traces identical:',result['sampled_trace_identical'],flush=True)


if __name__=='__main__':
    main()

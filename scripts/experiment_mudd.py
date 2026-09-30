"""Compile the small MUDD mixers while preserving their ordered BF16 sums."""
import json
from functools import partial
from pathlib import Path
import mlx.core as mx
from dwarf_violetto.runtime import load,prompt_tokens
from dwarf_violetto.limite import MuddMixer
from dwarf_violetto.bench import measure
from dwarf_violetto.sampling import make_sampler


def main():
    model,tokenizer,_=load('models/q8',optimized=True,fused=True)
    original=MuddMixer.combine
    compiled={(i,r):mx.compile(partial(original,model.model.mudd,layer_idx=i,r_way=r)) for i in [24,47] for r in [False,True]}
    active=False
    def combine(self,values,x_cur,layer_idx,r_way=False):
        if active and x_cur.shape[-2]==1:
            return compiled[(layer_idx,r_way)](values,x_cur)
        return original(self,values,x_cur,layer_idx,r_way)
    MuddMixer.combine=combine
    ids=prompt_tokens(tokenizer,'Find the last three digits of 1! + 2! + ... + 20!.')
    ids=(ids*20)[:1100]
    caches=[model.make_cache(),model.make_cache()]
    result={'runs':[]}
    try:
        for offset in range(0,len(ids),256):
            for mode,cache in zip([False,True],caches):
                active=mode;logits=model(mx.array([ids[offset:offset+256]]),cache);mx.eval(logits)
            mx.clear_cache()
        for token in ids[:256]:
            active=False;a=model(mx.array([[token]]),caches[0]);mx.eval(a)
            active=True;b=model(mx.array([[token]]),caches[1]);mx.eval(b)
            assert bool(mx.array_equal(a,b)),float(mx.max(mx.abs(a-b)))
        result['exact_decode_positions']=256
        del caches, logits, a, b
        mx.clear_cache()
        print('256 decode positions: exact logits',flush=True)
        sampler=make_sampler(.6,top_p=.95)
        for mode in [False,True]:
            active=mode;measure(model,ids[:128],128,512,sampler)
        for mode in [False,True,True,False,False,True,True,False]:
            active=mode;row=measure(model,ids[:128],512,512,sampler);row['mode']='mudd' if mode else 'baseline'
            result['runs'].append(row)
            Path('results/decode-mudd-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
            print(row['mode'],round(row['decode_tps'],2),flush=True)
        result['sampled_trace_identical']=all(r['tokens']==result['runs'][0]['tokens'] for r in result['runs'])
        Path('results/decode-mudd-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        MuddMixer.combine=original


if __name__=='__main__':
    main()

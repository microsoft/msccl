import math
ngpus = 8

nchunksperloop = 8
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="1" proto="LL" ngpus="{ngpus}" inplace="1" coll="allreduce" minBytes="8" maxBytes="262144">')

for i in range(ngpus):
    tbindex = 0
    print(f'  <gpu id="{i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{nchunksperloop}">')
    for j in range(ngpus):
        if i != j:
            print(f'    <tb id="{tbindex}" send="{j}" recv="{j}" chan="0">')
            step = 0
            if i != j:
                print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{j}" dstbuf="s" dstoff="{i}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                step += 1
                print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{i}" dstbuf="s" dstoff="{j}" cnt="1" depid="-1" deps="-1" hasdep="{1 if tbindex > 0 else 0}"/>')
                step += 1
            if tbindex == 0:
                for k in range(ngpus-3):
                    print(f'      <step s="{step}" type="nop" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0" cnt="1" depid="{k+1}" deps="1" hasdep="0"/>')
                    step += 1
                kCounter = 0
                for k in range(ngpus):
                    if k != i:
                        print(f'      <step s="{step}" type="re" srcbuf="s" srcoff="{k}" dstbuf="i" dstoff="{i}" cnt="1" depid="{ngpus-2 if kCounter == 0 else -1}" deps="{1 if kCounter == 0 else -1}" hasdep="{1 if kCounter == ngpus-2 else 0}"/>')
                        step += 1
                        kCounter += 1
            if i != j:
                print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{i}" dstbuf="i" dstoff="{i}" cnt="1" depid="{0}" deps="{2*ngpus-3}" hasdep="0"/>')
                step += 1
                print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{j}" dstbuf="i" dstoff="{j}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                step += 1
            print('    </tb>')
            tbindex+=1
    print('  </gpu>')
print('</algo>')

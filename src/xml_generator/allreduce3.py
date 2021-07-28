import math
ngpus = 8
instances = 1
nchunksperloop = instances*ngpus*ngpus
nnodes = 2
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="{instances}" proto="LL128">')
for node in range(nnodes):
    for i in range(ngpus):
        tbindex = 0
        print(f'  <gpu id="{node*ngpus+i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{nchunksperloop}">')
        for j in range(ngpus):
                for ch in range(instances):
                    if i != j:
                        print(f'    <tb id="{tbindex}" send="{node*ngpus+j}" recv="{node*ngpus+j}" chan="{ch}">')
                    else:
                        print(f'    <tb id="{tbindex}" send="-1" recv="-1" chan="0">')
                    step = 0
                    if i != j:
                        print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{(j*instances+ch)*ngpus}" dstbuf="s" dstoff="{(i*instances+ch)*ngpus}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="0"/>')
                        step += 1
                        print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{(i*instances+ch)*ngpus}" dstbuf="s" dstoff="{(j*instances+ch)*ngpus}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="1"/>')
                        step += 1
                    count = 0
                    for k in range(ngpus):
                        if k != i:
                            count += 1
                            print(f'      <step s="{step}" type="re" srcbuf="s" srcoff="{(k*instances+ch)*ngpus+j}" dstbuf="i" dstoff="{(i*instances+ch)*ngpus+j}" cnt="1" depid="{k}" deps="1" hasdep="{1 if count == ngpus-1 else 0}"/>')
                            step += 1

                    if i != j:
                        print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{(i*instances+ch)*ngpus}" dstbuf="i" dstoff="{(i*instances+ch)*ngpus}" cnt="{ngpus}" depid="{ngpus}" deps="{ngpus+1}" hasdep="0"/>')
                        step += 1
                        print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{(j*instances+ch)*ngpus}" dstbuf="i" dstoff="{(j*instances+ch)*ngpus}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="0"/>')
                        step += 1
                    print('    </tb>')
                    tbindex+=1
        ch = 0
        print(f'    <tb id="{tbindex}" send="{(node*ngpus+i+ngpus)%(ngpus*nnodes)}" recv="{(node*ngpus+i+ngpus)%(ngpus*nnodes)}" chan="{ch}">')
        step = 0
        for k in range(ngpus):
            print(f'      <step s="{step}" type="nop" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0" cnt="0" depid="{k}" deps="{ngpus if k != i else ngpus-2}" hasdep="0"/>')
            step += 1
        print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{(i*instances+ch)*ngpus}" dstbuf="i" dstoff="{(i*instances+ch)*ngpus}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="0"/>')
        step += 1
        print(f'      <step s="{step}" type="rrc" srcbuf="i" srcoff="{(i*instances+ch)*ngpus}" dstbuf="i" dstoff="{(i*instances+ch)*ngpus}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="1"/>')
        step += 1
        print('    </tb>')
        tbindex+=1
        print('  </gpu>')
print('</algo>')

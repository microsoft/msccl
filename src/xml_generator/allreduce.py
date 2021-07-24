import math
nchunksperloop = 16
instances = 1
ngpus = 16
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="{instances}" proto="LL">')

for i in range(ngpus):
    tbindex = 0
    print(f'  <gpu id="{i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{nchunksperloop}">')
    for j in range(ngpus):
        if i != j:
            for ch in range(instances):
                print(f'    <tb id="{tbindex}" send="{j}" recv="{j}" chan="{ch}">')
                step = 0
                print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{j*instances+ch}" dstbuf="s" dstoff="{i*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                step += 1
                for t in range(round(math.log2(ngpus))):
                    mask = 2**t
                    if mask == 1:
                        if tbindex % 2 == 0:
                            if tbindex == 0:
                                print(f'      <step s="{step}" type="rrc" srcbuf="i" srcoff="{i*instances+ch}" dstbuf="i" dstoff="{i*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="1"/>')
                            else:
                                print(f'      <step s="{step}" type="rrc" srcbuf="s" srcoff="{(tbindex-1)*instances+ch}" dstbuf="s" dstoff="{tbindex*instances+ch}" cnt="1" depid="{tbindex-1}" deps="{step}" hasdep="1"/>')
                        else:
                            print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{tbindex*instances+ch}" dstbuf="s" dstoff="{tbindex*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="1"/>')
                        step += 1
                    elif (tbindex % (2*mask)) == 0:
                        if tbindex == 0:
                            print(f'      <step s="{step}" type="re" srcbuf="s" srcoff="{(tbindex+mask)*instances+ch}" dstbuf="i" dstoff="{i*instances+ch}" cnt="1" depid="{tbindex+mask}" deps="{step-1}" hasdep="1"/>')
                        else:
                            print(f'      <step s="{step}" type="re" srcbuf="s" srcoff="{(tbindex+mask)*instances+ch}" dstbuf="s" dstoff="{tbindex*instances+ch}" cnt="1" depid="{tbindex+mask}" deps="{step-1}" hasdep="1"/>')
                        step += 1
                    
                print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{i*instances+ch}" dstbuf="i" dstoff="{i*instances+ch}" cnt="1" depid="0" deps="{round(math.log2(ngpus))}" hasdep="0"/>')
                step += 1
                print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{j*instances+ch}" dstbuf="i" dstoff="{j*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                step += 1
                print('    </tb>')
                tbindex+=1
    print('  </gpu>')
print('</algo>')

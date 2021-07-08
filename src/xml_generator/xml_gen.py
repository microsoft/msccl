nchunksperloop = 32
instances = 2
ngpus = 16
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, instances))

for i in range(ngpus):
    tbindex = 0
    nghr = (i+ngpus//2) % ngpus
    other_node_nghr = nghr
    print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(i, nchunksperloop, nchunksperloop, nchunksperloop))
    for ch in range(instances):
        print('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, nghr, ch))
        print('      <step s="{}" type="s" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(0, (nghr//(ngpus//2)) *(nchunksperloop//2) + ch*((nchunksperloop//2)//instances), ch*((nchunksperloop//2)//instances), (nchunksperloop//2)//instances))
        print('    </tb>')
        tbindex+=1
    for ch in range(instances):
        print('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, nghr, ch))
        print('      <step s="{}" type="r" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(0, (nghr//(ngpus//2)) *(nchunksperloop//2) + ch*((nchunksperloop//2)//instances), ch*((nchunksperloop//2)//instances), (nchunksperloop//2)//instances))
        print('    </tb>')
        tbindex+=1
    if True:
      for j in range(ngpus//2):
        nghr = j+(ngpus//2)*(i//(ngpus//2))
        nghr_nghr = (nghr+ngpus//2) % ngpus
        if i == nghr:
           for ch in range(instances):
               print('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex))
               print('      <step s="{}" type="cpy" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(0, i * instances+ch, i * instances+ch, 1))
               print('      <step s="{}" type="cpy" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="0" hasdep="0"/>'.format(1, j*instances+ch, other_node_nghr*instances+ch, 1, instances + j // (ngpus//2//instances)))
               print('    </tb>')
               tbindex+=1
        else:
            for ch in range(instances):
                print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbindex, nghr, nghr, ch))
                print('      <step s="{}" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(0, nghr*instances+ch, i*instances+ch, 1))
                print('      <step s="{}" type="r" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(1, i*instances+ch, nghr*instances+ch, 1))
                print('      <step s="{}" type="s" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="0" hasdep="0"/>'.format(2, j*instances+ch, nghr_nghr*instances+ch, 1, instances + j//(ngpus//2//instances)))
                print('      <step s="{}" type="r" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="0" hasdep="0"/>'.format(3, i*instances+ch, nghr_nghr*instances+ch, 1))
                print('    </tb>')
                tbindex+=1
    print('  </gpu>')
print('</algo>')


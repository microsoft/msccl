nnodes = 3
ngpuspernode = 2
instances = 1
nchunksperloop = nnodes*ngpuspernode*instances
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, instances))

def CrossNodeNghr(node, g):
    return g if node > g else g+1
for node in range(nnodes):
    for g in range(ngpuspernode):
        tbindex = 0
        crossnodenghr = CrossNodeNghr(node,g)
        print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(node*ngpuspernode+g, nchunksperloop, nchunksperloop, nchunksperloop))
        for ch in range(instances):
            print('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="s" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(crossnodenghr, 0, ngpuspernode))
            print('      <step s="1" type="s" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(crossnodenghr, ngpuspernode, ngpuspernode*(ngpuspernode-1)))
            print('    </tb>')
            tbindex+=1
        for ch in range(instances):
            print('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="r" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(node, 0, ngpuspernode))
            print('      <step s="1" type="r" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(crossnodenghr, ngpuspernode, ngpuspernode*(ngpuspernode-1)))
            print('    </tb>')
            tbindex+=1
        for i in range(ngpuspernode):
            withinnodenghr = node*ngpuspernode+(i if g > i else i+1)
            withinnodenghrscrossnodenghr = CrossNodeNghr(node, withinnodenghr)
            for ch in range(instances):
                print('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex))
                print('      <step s="0" type="cpy" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(node*ngpuspernode+g, node*ngpuspernode+g, 1))
                step = 1
                for j in range(ngpuspernode):
                    if j != g:
                        print('      <step s="{}" type="nop" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="0" depid="{}" deps="{}" hasdep="{}"/>'.format(step, j+2, 2, 1 if j == ngpuspernode-1 else 0))
                        step += 1
                print('    </tb>')
                tbindex+=1
            else:
                for ch in range(instances):
                    print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbindex, withinnodenghr, withinnodenghr, ch))
                    print('      <step s="0" type="s" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(withinnodenghrscrossnodenghr, g*ngpuspernode, ngpuspernode))
                    print('      <step s="1" type="r" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(crossnodenghr, i*ngpuspernode, ngpuspernode))
                    print('      <step s="2" type="s" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="0" hasdep="0"/>'.format(g*ngpuspernode, crossnodenghr*ngpuspernode, ngpuspernode))
                    print('      <step s="3" type="r" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="0" hasdep="0"/>'.format(i*ngpuspernode, withinnodenghrscrossnodenghr*ngpuspernode, ngpuspernode))
                    print('    </tb>')
                    tbindex+=1
        print('  </gpu>')

print('</algo>')


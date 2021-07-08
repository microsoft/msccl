nnodes = 3
ngpuspernode = 2
instances = 1
nchunksperloop = nnodes*ngpuspernode*instances
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, instances))

def CrossNodeNghr(node, g):
    nghrNode = g if node > g else g+1
    nghrG = node if nghrNode > node else node-1
    return nghrNode, nghrG, nghrNode * ngpuspernode + nghrG
for node in range(nnodes):
    for g in range(ngpuspernode):
        tbindex = 0
        nghrNode, nghrG, crossnodenghr = CrossNodeNghr(node,g)
        print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(node*ngpuspernode+g, nchunksperloop, nchunksperloop, 2*ngpuspernode**2))
        for ch in range(instances):
            print('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="s" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(0, ngpuspernode**2, ngpuspernode**2, 2+g, ngpuspernode-1))
            print('    </tb>')
            tbindex+=1
        for ch in range(instances):
            print('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="r" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(0, ngpuspernode**2, ngpuspernode**2, 2+g, ngpuspernode-1))
            print('    </tb>')
            tbindex+=1
        for withinnodenghr  in range(ngpuspernode):
            withinNghrNode, withinNghrG, withinCrossNodeNghr = CrossNodeNghr(node, withinnodenghr)
            if withinnodenghr == g:
                for ch in range(instances):
                    print('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex))
                    print('      <step s="0" type="cpy" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(nghrNode*ngpuspernode, g*ngpuspernode, 1))
                    step = 1
                    for j in range(ngpuspernode):
                        if j != g:
                            print('      <step s="{}" type="nop" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="0" depid="{}" deps="{}" hasdep="{}"/>'.format(step, j+2, 1, 1 if j == ngpuspernode-1 else 0))
                            step += 1
                    print('      <step s="{}" type="cpy" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(step, node*ngpuspernode+g, node*ngpuspernode+g, 1))
                    step += 1
                    print('    </tb>')
                    tbindex+=1
            else:
                for ch in range(instances):
                    print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbindex, withinnodenghr, withinnodenghr, ch))
                    print('      <step s="0" type="s" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(withinNghrNode*ngpuspernode, g*ngpuspernode, ngpuspernode))
                    print('      <step s="1" type="r" srcbuf="i" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(nghrNode*ngpuspernode, withinnodenghr*ngpuspernode, ngpuspernode))
                    print('      <step s="2" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(node*ngpuspernode+withinnodenghr, node*ngpuspernode+g, 1))
                    print('      <step s="3" type="r" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(node*ngpuspernode+g, node*ngpuspernode+withinnodenghr, 1))
                    step = 4
                    for i in range(ngpuspernode):
                        print('      <step s="{}" type="s" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(step, ngpuspernode**2+j*ngpuspernode+withinnodenghr, nghrNode*ngpuspernode+j, 1, 1 if i == 0 else -1, 0 if i == 0 else -1))
                        step += 1
                        print('      <step s="{}" type="r" srcbuf="s" srcoff="{}" dstbuf="o" dstoff="{}" cnt="{}" depid="{}" deps="{}" hasdep="0"/>'.format(step, ngpuspernode**2+j*ngpuspernode+g, nghrNode*ngpuspernode+j, 1, 1 if i == 0 else -1, 0 if i == 0 else -1))
                        step += 1
                    print('    </tb>')
                    tbindex+=1
        print('  </gpu>')

print('</algo>')


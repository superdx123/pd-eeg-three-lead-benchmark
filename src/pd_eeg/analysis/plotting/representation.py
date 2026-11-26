"""Input- and parameter-matched frequency-band representation comparison."""

from itertools import combinations
import numpy as np
from . import charts as core

def draw():
    plt=core.setup()
    from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
    import numpy as np
    p=core.PALETTE
    fig=plt.figure(figsize=(core.WIDTH,4.3),facecolor='white')
    canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off()
    flow=fig.add_axes([0,0,1,1]);flow.set(xlim=(0,7.10),ylim=(0,4.30));flow.axis('off')
    flow.set_zorder(0)
    labels=[];nodes=[];edges=[]
    def text(x,y,s,size=9.5,bold=False,ha='center',box=None):
        size = max(8.9, size * .92)
        item=flow.text(x,y,s,ha=ha,va='center',fontsize=size,family=core.SERIF_FONT,
                       weight='bold' if bold else 'normal',color=p['ink'],linespacing=1.12,zorder=5)
        labels.append((item,box));return item
    def node(key,x,cy,w,h,s,face=None,bold=False,size=9.5):
        flow.add_patch(FancyBboxPatch((x,cy-h/2),w,h,boxstyle='round,pad=0,rounding_size=.025',
                          facecolor=face or p['flow_light_blue'],edgecolor=p['flow_blue_outline'],lw=.8,zorder=2))
        box=[x,cy-h/2,x+w,cy+h/2]
        text(x+w/2,cy,s,size,bold,box=box);nodes.append({'key':key,'label':s,'bounds':box})
    def route(points,arrow=True):
        if len(points)>2:flow.plot(*zip(*points[:-1]),color='#606060',lw=.75,zorder=1)
        if arrow:
            flow.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=7.6,
                           color='#606060',lw=.75,shrinkA=0,shrinkB=0,zorder=1))
        else:flow.plot(*zip(*points),color='#606060',lw=.75,zorder=1)
        edges.append({'points':points,'arrow':arrow})
    def triplets(x,cy,band_expanded):
        for j in range(6):
            face=p['flow_coral'] if not band_expanded or j==0 else p['flow_blue']
            left=x+j*.145
            flow.add_patch(Rectangle((left,cy-.16),.112,.32,facecolor=face,
                          edgecolor=p['flow_blue_outline'],lw=.5,zorder=2))
            for k in (1,2):flow.plot([left,left+.112],[cy-.16+k*.32/3]*2,
                                     color=p['flow_blue_outline'],lw=.3,zorder=3)

    text(.2485,4.1495,'(a)  Input and parameter-matched comparison',11.1,True,ha='left')
    flow.add_patch(Rectangle((.15,1.69),6.80,2.28,facecolor='none',edgecolor='#777777',
                            lw=.65,linestyle=(0,(3.2,2.4)),zorder=0))
    text(.73,3.67,'Standardized input',9.7,True)
    text(.73,3.47,'C3 / Cz / C4',9.3)
    t=np.linspace(0,1,140)
    for j in range(3):
        y=3.29-j*.16
        z=.030*np.sin(2*np.pi*(3+j)*t)+.012*np.sin(2*np.pi*(8+j)*t+.4*j)
        flow.plot(.35+.76*t,y+z,color=p['flow_blue_outline'],lw=.65,zorder=2)
    text(.73,2.70,'3 × 512',9.5)
    route([(1.15,3.13),(1.47,3.13)],False)
    route([(1.47,2.45),(1.47,3.45)],False)
    flow.plot(1.47,3.13,'o',color='#606060',ms=2.1,zorder=3)
    text(2.35,3.79,'full',9.8,True)
    text(2.35,2.79,'R18',9.8,True)
    node('full_construction',1.77,3.45,1.16,.43,'Raw + five bands',p['flow_blue'],size=9.3)
    node('r18_construction',1.77,2.45,1.16,.43,'Raw triplet × 6',p['flow_blue'],size=9.3)
    for cy in (3.45,2.45):route([(1.47,cy),(1.73,cy)])
    triplets(3.31,3.45,True);triplets(3.31,2.45,False)
    for cy in (3.45,2.45):
        route([(2.97,cy),(3.27,cy)])
        text(3.73,cy-.30,'18 × 512',9.4)
        route([(4.18,cy),(4.66,cy)])
    text(5.32,3.82,'Independent training',9.5,True)
    node('full_model',4.70,3.45,1.24,.52,'Multi-scale CNN\nbackbone',p['flow_light_blue'],size=9.7)
    node('r18_model',4.70,2.45,1.24,.52,'Multi-scale CNN\nbackbone',p['flow_light_blue'],size=9.7)
    for cy in (3.45,2.45):
        route([(5.98,cy),(6.25,cy)])
        for k in range(2):
            flow.add_patch(Rectangle((6.30+k*.18,cy-.09),.18,.18,
                           facecolor=p['flow_coral'],edgecolor=p['flow_coral_outline'],lw=.55,zorder=2))
        text(6.48,cy-.30,'P(PD)',9.4)
    text(3.55,1.96,'Same architecture · 67,190 nominal trainable parameters per model',9.6)
    text(3.55,1.78,'Five subject-level folds · seeds 11, 22, 33 · 50 epochs',9.4)

    canvas.text(.035,.355,'(b)  Paired differences: full − R18',fontsize=core.TITLE,fontweight='bold',va='center')
    ax=fig.add_axes([.195,.108,.395,.185]);scale={'limits':(-.12,.16),'ticks':(-.1,0,.1),'reference':0}
    core.axis(ax,scale,2)
    estimate=fig.add_axes([.625,.108,.340,.185]);estimate.set(xlim=(0,1),ylim=ax.get_ylim());estimate.set_axis_off()
    records=core.load(core.DATA / 'representation.json')['paired_differences']
    for y,r in enumerate(records):
        name='ΔAUC' if y==0 else 'Δ balanced\naccuracy'
        ax.text(-.040,y,name,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=core.TICK)
        core.point(ax,r['point'],r['ci95'],y,core.FOCAL,'D',core.FOCAL_PT)
        estimate.text(.02,y,f"{core.label(r['point'],signed=True)}  [{core.label(r['ci95'][0])}, {core.label(r['ci95'][1])}]",
                      ha='left',va='center',fontsize=core.TICK)
    canvas.text(.625,.316,'Difference [95% conditional interval]',ha='left',va='center',fontsize=core.TICK,color=core.GRAY)
    canvas.text(.558,.027,'Identical input dimensions and backbone specification; matched training budget.',
                ha='center',va='center',fontsize=core.TICK,color=core.GRAY)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();inverse=flow.transData.inverted()
    node_text_bounds=[]
    for a,box in labels:
        b=a.get_window_extent(renderer).transformed(inverse)
        assert b.x0>.04 and b.x1<7.06 and b.y0>1.70 and b.y1<4.26,('FLOW_BOUNDS',a.get_text())
        if box:
            l,bot,r,top=box
            assert l+.02<=b.x0 and b.x1<=r-.02 and bot+.015<=b.y0 and b.y1<=top-.015,('NODE_TEXT',a.get_text())
            node_text_bounds.append({'label':a.get_text(),'font_pt':a.get_fontsize(),
                                     'box_inches':box,'text_inches':[b.x0,b.y0,b.x1,b.y1]})
    return core.export(plt,fig,'representation_control')

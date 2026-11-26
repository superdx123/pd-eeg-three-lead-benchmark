"""Consistent manuscript plots from published aggregate estimates."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path
import csv
import argparse
import hashlib
import json
import os

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path("figures")
FORMATS = ("pdf", "svg", "png")
STYLE_SPEC = json.loads((DATA / "style.json").read_text(encoding="utf-8"))
PALETTE = STYLE_SPEC["palette"]
SERIF_FONT = "DejaVu Serif"

ORDER = ('BE-MSNet','EEGConformer','EEGNeX','ATCNet','FBCNet','EEGNet',
         '4L-CNN','Lee-CRNN','PSD-AE-RBFNN','RBF-SVM','TDCNet†','Spectral LR')
LABEL = {'BE-MSNet':'BE-MSNet','4L-CNN':'4L-CNN','Lee-CRNN':'Lee-CRNN','PSD-AE-RBFNN':'PSD-AE-RBFNN','RBF-SVM':'RBF-SVM',
         'TDCNet†':'TDCNet†','Spectral LR':'Spectral LR','Harmonized LR':'Harmonized LR'}
WIDTH,DPI = 7.10,600
FONT,TICK,TITLE = 9.7,9.2,11.1
BLUE = PALETTE['chart_comparator_line']
FOCAL,ORANGE = PALETTE['chart_focal_line'],PALETTE['chart_comparator_line']
GRAY,INK = PALETTE['secondary_text'],PALETTE['ink']
GRID,RULE,REF = PALETTE['grid'],PALETTE['axis'],PALETTE['reference']
PALE,STRIPE = PALETTE['flow_light_blue'],PALETTE['white']
AXIS_LW,GRID_LW,REF_LW,TICK_LW,TICK_LEN = .7,.55,.95,.7,3.
CI_LW,CAP_PT,CAP_LW,FOCAL_PT,OTHER_PT = 1.12,4.9,.85,5.3,4.5
COLS,AX_WIDTH,EST_GAP,EST_WIDTH = (.190,.605),.286,.015,.079
ABS = {'limits':(-.035,1.035),'ticks':(0,.2,.4,.6,.8,1),'reference':.5}
PAIR = {'limits':(-.235,.675),'ticks':(-.2,0,.2,.4,.6),'reference':0}
SCORES = ('auc','balanced_accuracy')
STYLE = {'width_inches':WIDTH,'font_pt':[FONT,TICK,TITLE],
         'color':{'BE-MSNet':FOCAL,'BE-MSNet_fill':PALETTE['chart_focal_fill'],
                  'comparator':BLUE,'comparator_fill':PALETTE['chart_comparator_fill'],
                  'S':FOCAL,'W':ORANGE,'ink':INK,'grid':GRID,'rule':RULE},
         'axis_lw_pt':AXIS_LW,'grid_lw_pt':GRID_LW,'tick_lw_pt':TICK_LW,
         'tick_length_pt':TICK_LEN,'reference_lw_pt':REF_LW,'ci_lw_pt':CI_LW,
         'visible_spines':['bottom'],'tick_decimals':1,'estimate_decimals':4,
         'background':'white','row_shading':False,
         'style_spec_sha256':hashlib.sha256((DATA/'style.json').read_bytes()).hexdigest()}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def label(value,signed=False):
    d=Decimal(f'{value:.12f}').quantize(Decimal('.0001'),rounding=ROUND_HALF_UP)
    return format(d,'+.4f' if signed else '.4f')


def load(name): return json.loads(Path(name).read_text(encoding='utf-8'))


def setup():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    family = "DejaVu Sans"
    global SERIF_FONT
    SERIF_FONT = "DejaVu Serif"
    plt.rcParams.update({'font.family':family,'font.size':FONT,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'svg.hashsalt':'pd-eeg-benchmark','figure.dpi':120,'savefig.dpi':DPI,
                         'text.color':INK,'xtick.color':INK,'ytick.color':INK,
                         'axes.labelcolor':INK,'savefig.facecolor':'white','figure.facecolor':'white'})
    return plt


def axis(ax,scale,n):
    ax.set(xlim=scale['limits'],ylim=(n-.46,-.55))
    ax.set_xticks(scale['ticks'],[f'{x:.1f}' for x in scale['ticks']]);ax.set_yticks([])
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color(RULE);ax.spines['bottom'].set_linewidth(AXIS_LW)
    ax.spines['bottom'].set_bounds(scale['ticks'][0],scale['ticks'][-1])
    ax.tick_params(axis='x',pad=3.5,length=TICK_LEN,width=TICK_LW,color=RULE,labelsize=TICK)
    ax.set_axisbelow(True);ax.grid(axis='x',color=GRID,linewidth=GRID_LW)
    ax.axvline(scale['reference'],color=REF,lw=REF_LW,ls=(0,(2.6,2.4)),zorder=1.4)


def point(ax,point,bounds,y,color,shape,size):
    lo,hi=map(float,bounds)
    bar,=ax.plot([lo,hi],[y,y],color=color,lw=CI_LW,zorder=2.2)
    caps,=ax.plot([lo,hi],[y,y],marker='|',ms=CAP_PT,mew=CAP_LW,ls='none',color=color,zorder=2.2)
    face=PALETTE['chart_focal_fill'] if color==FOCAL else PALETTE['chart_comparator_fill']
    dot,=ax.plot([point],[y],marker=shape,ms=size,ls='none',color=color,mfc=face,mec=color,mew=.65,zorder=3)
    assert float(dot.get_xdata()[0])==point and list(map(float,bar.get_xdata()))==[lo,hi]
    assert list(map(float,caps.get_xdata()))==[lo,hi]
    for v,radius in ((point,size/2),(lo,CAP_LW/2),(hi,CAP_LW/2)):
        x=ax.transData.transform((v,y))[0];px=radius*ax.figure.dpi/72
        assert x-px>ax.bbox.x0 and x+px<ax.bbox.x1,'CLIPPED_MARKER_OR_CI'


def export(plt,fig,stem):
    OUT.mkdir(parents=True, exist_ok=True)
    from matplotlib.text import Text
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert boxes
    for item,rect in boxes:
        assert rect.x0>=-.25 and rect.y0>=-.25 and rect.x1<=fig.bbox.width+.25 and rect.y1<=fig.bbox.height+.25,('CLIPPED_TEXT',stem,item.get_text())
    for (a,p),(b,q) in combinations(boxes,2):
        assert not(min(p.x1,q.x1)-max(p.x0,q.x0)>2 and min(p.y1,q.y1)-max(p.y0,q.y0)>2),('OVERLAPPING_LABELS',stem,a.get_text(),b.get_text())
    minfont=min(t.get_fontsize() for t,_ in boxes)*174/(WIDTH*25.4)
    assert minfont>=8.5,('FONT_TOO_SMALL',stem,minfont)
    files={}
    for ext in FORMATS:
        path=OUT/f'{stem}.{ext}'
        assert path.resolve().parent==OUT.resolve()
        kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        fig.savefig(path,dpi=DPI,**kw)
        files[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    plt.close(fig)
    return {'files':files,'minimum_final_font_pt':minfont,'style':STYLE,
            'text_labels':sorted(t.get_text() for t,_ in boxes),
            'all_text_inside_canvas':True,'all_label_boxes_nonoverlapping':True}


def title(plt,fig,col,text,y,rule_y):
    left=COLS[col];right=left+AX_WIDTH+EST_GAP+EST_WIDTH
    fig.text((left+right)/2,y,text,color=INK,fontweight='bold',ha='center',va='center',fontsize=TITLE)
    fig.add_artist(plt.Line2D([left,right],[rule_y,rule_y],transform=fig.transFigure,lw=.65,color=GRID))


def render_forest(plt,stem,panels,headers,foot):
    paired=stem=='s_paired_differences';scale=PAIR if paired else ABS
    n=len(panels[0]['records']);dense=n>3
    fig=plt.figure(figsize=(WIDTH,6.18 if dense else 3.62),facecolor='white')
    for col,head in enumerate(headers):title(plt,fig,col,head,.954 if dense else .943,.924 if dense else .898)
    for i,panel in enumerate(panels):
        row,col=divmod(i,2);y0=((.550,.121) if dense else (.565,.166))[row]
        height=.330 if dense else .255;left=COLS[col]
        ax=fig.add_axes([left,y0,AX_WIDTH,height]);est=fig.add_axes([left+AX_WIDTH+EST_GAP,y0,EST_WIDTH,height])
        axis(ax,scale,n);est.set(xlim=(0,1),ylim=ax.get_ylim());est.set_axis_off()
        metric=('ΔAUC' if row==0 else 'Δ balanced accuracy') if paired else ('AUC' if row==0 else 'Balanced accuracy')
        heading=1.039 if dense else 1.080
        ax.text(0,heading,f"({panel['panel']})  {metric}",transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=FONT)
        est.text(1,heading,'Difference' if paired else 'Estimate',transform=est.transAxes,ha='right',va='bottom',fontsize=TICK,color=GRAY)
        for y,record in enumerate(panel['records']):
            model=record['model'];focal=model=='BE-MSNet'
            color,shape=(FOCAL,'D') if paired else (FOCAL,'o') if focal else (BLUE,'s')
            if col==0:
                ax.text(-.032,y,LABEL.get(model,model),transform=ax.get_yaxis_transform(),ha='right',va='center',
                        color=INK,fontweight='bold' if focal else 'normal')
            point(ax,record['point'],record['ci95'],y,color,shape,FOCAL_PT if focal else OTHER_PT)
            record['estimate_label']=label(record['point'],signed=paired)
            est.text(.96,y,record['estimate_label'],ha='right',va='center',fontsize=TICK,
                     color=INK,fontweight='bold' if focal else 'normal')
    fig.text(.558,.038,foot,ha='center',va='center',fontsize=TICK,color=GRAY)
    return {**export(plt,fig,stem),'panels':panels,'axis_limits':list(scale['limits']),
            'labeled_ticks':list(scale['ticks']),'reference_line':scale['reference']}


def forest_panels(accepted,stem,new):
    original=accepted['figures'][stem]['records'];panels=[]
    for ri,metric in enumerate(SCORES):
        for ci,cohort in enumerate(('778','490')):
            records=deepcopy([r for r in original if r['cohort']==cohort and r['metric']==metric])
            for r in records:
                before=deepcopy(r)
                if cohort=='778' and (stem=='s_paired_differences' or r['model']=='BE-MSNet'):
                    if stem=='s_paired_differences':
                        v=next(x for x in new['paired_subject_independent'] if x['baseline']==r['model'] and x['metric']==metric)
                        r.update(point=v['difference'],ci95=v['ci95'])
                    else:
                        v=new['subject_independent' if stem=='s_absolute_intervals' else 'within_recording'][metric]
                        r.update(point=v['point'],ci95=v['ci95'])
                else:
                    assert r==before
            panels.append({'panel':'abcd'[ri*2+ci],'metric':metric,'cohort':cohort,'records':records})
    return panels


def render_ws(plt,accepted,new):
    from matplotlib.lines import Line2D
    records=deepcopy(accepted['figures']['within_recording_vs_subject_independent']['records']);fig=plt.figure(figsize=(WIDTH,2.48))
    for i,metric in enumerate(SCORES):
        ax=fig.add_axes([COLS[i],.237,AX_WIDTH+EST_GAP+EST_WIDTH,.443]);axis(ax,ABS,2);ax.set_ylim(1.66,-.64)
        ax.text(0,1.08,f"({'ab'[i]})  "+('AUC' if i==0 else 'Balanced accuracy'),transform=ax.transAxes,
                ha='left',va='bottom',fontweight='bold',fontsize=FONT)
        for y,(cohort,name) in enumerate(zip(('778','490'),('ds002778','ds003490 (EO)'))):
            r=next(x for x in records if x['cohort']==cohort and x['metric']==metric)
            if cohort=='778':
                s=new['subject_independent'][metric]['point'];w=new['within_recording'][metric]['point']
                r.update(S=s,W=w,descriptive_W_minus_S=w-s)
            else:s,w=r['S'],r['W']
            delta=r['descriptive_W_minus_S']
            if i==0:ax.text(-.032,y,name,transform=ax.get_yaxis_transform(),ha='right',va='center')
            ax.plot([s,w],[y,y],color=RULE,lw=CI_LW,ls=(0,(1.4,2.2)),zorder=2)
            ax.plot(s,y,'o',color=FOCAL,mfc=PALETTE['chart_focal_fill'],ms=FOCAL_PT,zorder=3,mec=FOCAL,mew=.65)
            ax.plot(w,y,'s',color=ORANGE,mfc=PALETTE['chart_comparator_fill'],ms=FOCAL_PT,zorder=3,mec=ORANGE,mew=.65)
            ax.annotate(label(s),(s,y),xytext=(-1,7),textcoords='offset points',ha='right',va='bottom',color=INK,fontsize=TICK)
            ax.annotate(label(w),(w,y),xytext=(0,-7),textcoords='offset points',ha='right',va='top',color=INK,fontsize=TICK)
            ax.text(.05,y,f'Δ {label(delta,signed=True)}',ha='left',va='center',fontsize=TICK)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=FOCAL,ms=FOCAL_PT,label='S: Unseen subjects'),
                        Line2D([],[],marker='s',ls='',color=ORANGE,ms=FOCAL_PT,label='W: Within-recording windows')],
               loc='upper center',bbox_to_anchor=(.557,.971),ncol=2,frameon=False,fontsize=FONT,
               handletextpad=.4,columnspacing=1.3,borderpad=0)
    fig.text(.558,.039,'Descriptive comparison: Δ = W - S. Dotted connectors are not interval estimates.',
             ha='center',va='center',fontsize=TICK,color=GRAY)
    return {**export(plt,fig,'within_recording_vs_subject_independent'),'records':records,'axis_limits':list(ABS['limits']),
            'confidence_intervals_drawn':False}


def probability_bins(plt,new):
    source_names=('S_490','transfer_778_to_490','transfer_490_to_778')
    unified=new['calibration']
    panels=[{'panel':n,'bins':unified[n]['bins'],'source':'new_primary_S_778' if n=='S_778' else 'unchanged_original_probability_bins'}
            for n in ('S_778',*source_names)]
    titles=('ds002778: S','ds003490 (EO): S','ds002778 to ds003490 (EO)','ds003490 (EO) to ds002778')
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,6.2))
    fig.subplots_adjust(left=.105,right=.985,bottom=.105,top=.885,wspace=.31,hspace=.55)
    for i,(ax,panel,title_text) in enumerate(zip(axes.flat,panels,titles)):
        values=[v for v in panel['bins'] if v['count']]
        assert all(v['mean_predicted_probability'] is not None and v['observed_positive_fraction'] is not None for v in values)
        ax.set(xlim=(-.035,1.035),ylim=(-.075,1.105),xticks=[0,.2,.4,.6,.8,1],yticks=[0,.2,.4,.6,.8,1])
        ax.plot([0,1],[0,1],color=REF,lw=REF_LW,ls=(0,(2.6,2.4)),zorder=1)
        ax.plot([v['mean_predicted_probability'] for v in values],[v['observed_positive_fraction'] for v in values],
                color=FOCAL,marker='o',mfc=PALETTE['chart_focal_fill'],ms=4.2,lw=1.05,zorder=3)
        for v in values:
            x,y=v['mean_predicted_probability'],v['observed_positive_fraction']
            ax.annotate(f"n={v['count']}",(x,y),xytext=(0,7 if y<=.88 else -14),textcoords='offset points',
                        ha='center',fontsize=TICK,color=INK,clip_on=False)
        ax.set_title(title_text,pad=10,fontsize=10.4,color=INK)
        ax.text(-.015,1.15,'('+'abcd'[i]+')',transform=ax.transAxes,fontsize=10.8,fontweight='bold',color=INK)
        ax.set_xlabel('Mean predicted PD probability',labelpad=5,fontsize=FONT)
        if i%2==0:ax.set_ylabel('Observed PD fraction',labelpad=7,fontsize=FONT)
        ax.spines[['top','right']].set_visible(False)
        for side in ('bottom','left'):ax.spines[side].set_color(RULE);ax.spines[side].set_linewidth(AXIS_LW)
        ax.tick_params(length=TICK_LEN,width=TICK_LW,color=RULE,labelsize=TICK)
        ax.set_axisbelow(True);ax.grid(color=GRID,linewidth=GRID_LW)
    return {**export(plt,fig,'probability_bins'),'panels':panels,
            'empty_bins_omitted':True,'count_labels_shown':True,'recalibration_performed':False, 'recalibration': False}

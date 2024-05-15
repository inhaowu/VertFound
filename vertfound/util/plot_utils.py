"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs

def GetConfusionMatrix(filename):
    '''Reads an Excel file containing a confusion matrix
    Parameters
    ----------
    filename : str
        The name of the file. 
        It has to be an excel file with the following structure:
            - The first row indicates the names of the estimated classes
            - The first column indicates the names of the actual classes
            - The remaining elemnts contains de confusion matrix

    Returns
    -------
    cm : confusion matrix of dimension (C,C)
    cl: Vector of dimension C
        The labels of the classes
    '''    
    df = pd.read_excel(filename, index_col=0)
    cm = df.values
    cl = df.columns.values
    return cm,cl

def GetConfusionStar(cm,balanced):
    '''Obtains the structure of the confusion star (or a confusion gear)
    Parameters
    ----------
    cm : confusion matrix of dimension (C,C)
    balanced: bool (optional).
        A flag used to indicate if the balanced version of the
        confusion star has to be drawn

    Returns
    -------
    em : error matrix of dimension (C,C-1)
        It is a confusion matrix with redundancies (diagonal) removed
    th : Vector of dimension C·(C-1)
        Angles of the sectors in the radial plot
    beta : Vector of dimension C
        Angles where starts each actual class in the radial plot
    '''    
    C = cm.shape[0] #Number of classes
    um = cm2um(cm,unit=True) #Convert to unit matrix if required
    em = um2em(um) #Convert to error matrix
    
    #Compute beta: the angle where starts each actual class
    if balanced:
        beta = np.cumsum(2*np.pi/C*np.ones(C)) #Equally spaced classes
    else:
        m = np.sum(cm,axis=1) #Number of instances per class
        M = np.sum(m) #Total number of instances
        beta = np.cumsum(2*np.pi*m/M) #Proportional to #of instances per class
    beta = np.insert(beta,0,0)

    #Compute th: the angle where starts each index in each actual class
    nth = (C-1)*C
    th = np.zeros(nth)
    for k in np.arange(nth):
        i = int(k/(C-1))
        j = k-i*(C-1)
        dth = (beta[i+1]-beta[i]) / (C-1)
        th[k] = beta[i]+j*dth
    return em,th,beta

def PlotConfusionStar(cm,cl,star=True,balanced=True,log=False,fill=True,
                      edgecolor=None,outerlabel=True,innerlabel=True,
                      rotoutlabel=False,rotinnlabel=False, name='star.pdf'):
    '''Plot a confusion star (or a confusion gear).
    Parameters
    ----------
    cm : confusion matrix of dimension (C,C)
    cl: Vector of dimension C
        The labels of the classes
    star: bool (optional). A flag used to indicate the type of plot
        if True it plots a confusion star
        if False it plots a confusion gear
    balanced: bool (optional).
        A flag used to indicate if the balanced version of the
        confusion star has to be drawn
    log: bool (optional). A flag used to indicate the scale of the plot.
        It only applies to confusion star plots (not to confusion gear plots)
        if True it uses log scale
        if False it uses linear scale
    fill: bool (optional).
        If True the sectors are filled
    edgecolor: color specification or None.
        Star edge color
    outerlabel: bool (optional).
        If True the label of actual classes are drawn
    innerlabel: bool (optional).
        If True the label of indices to estimated classes are drawn
    rotoutlabel: bool (optional).
        If True the label of actual classes are rotated
    rotinnlabel: bool (optional).
        If True the label of indices to estimated classes are rotated
    '''    
    em,th,beta = GetConfusionStar(cm,balanced)
    PlotConfusionGrid(em,th,cl,star,log,outerlabel,innerlabel,
                      rotoutlabel,rotinnlabel)
    sectors = GetConfusionSectors(em,th,beta,star,log,fill)
    PlotConfusionSectors(sectors,edgecolor)

    plt.savefig(name)

def PlotConfusionGrid(em,th,cl,star,log,outerlabel,innerlabel,
                      rotoutlabel,rotinnlabel):
    '''Draw the grid of the confusion star (or a confusion gear)
        This grid includes:
            - The radial grid (circles)
            - A text with the values (scale) corresponding to each circle
            - The angular grid (radii)
            - A text with the class associated to each sector 
                (drawn in the outermost circle)
            - A text with the class associated to each actual class 
                (drawn in the outermost circle, beyond the sector text)
            
    Parameters
    ----------
    em : error matrix of dimension (C,C-1)
        It is a confusion matrix with redundancies (diagonal) removed
    th : Vector of dimension C·(C-1)
        Angles of the sectors in the radial plot
    cl: Vector of dimension C
        The labels of the classes
    star: bool. A flag used to indicate the type of plot
        if True it plots a confusion star
        if False it plots a confusion gear
    log: bool. A flag used to indicate the scale of the plot.
        It only applies to confusion star plots (not to confusion gear plots)
        if True it uses log scale
        if False it uses linear scale
    outerlabel: bool.
        If True the label of actual classes are drawn
    innerlabel: bool.
        If True the label of indices to estimated classes are drawn
    rotoutlabel: bool.
        If True the label of actual classes are rotated
    rotinnlabel: bool.
        If True the label of indices to estimated classes are rotated
    '''    
    #Compute radial grid: position (rgv), position of the text (rgvt) &
                        # format of the text (FormTgrid)
    if star: #Grid for the confusion star
        r = em #Radii vector
        if log: #Confusion star with log scale
            rmin = 0.01
            r = np.log10(np.maximum(rmin,r))-np.log10(rmin)
            rmax = 4
            rgvt = np.asarray([0.1,1,10,100])
            rgv = np.log10(np.maximum(rmin,rgvt))-np.log10(rmin)
            FormTgrid = ['{:2.1f}','{:3.0f}','{:3.0f}','{:3.0f}']
        else: #Confusion star with linear scale
            rmax = np.max(r)
            rmax = 4*np.ceil(rmax/4)
            rgvt = rmax/4*np.asarray([1,2,3,4])
            rgv = rgvt
            FormTgrid = ['{:3.0f}']*4
    else: #Grid for the confusion gear (always linear scale)
        r = 100-em
        rmax = 100
        rgvt = np.asarray([25,50,75,100])
        rgv = rgvt
        FormTgrid = ['{:3.0f}']*4

    #Plot the angular grid (radii)
    C = em.shape[0]
    for i in range(C):
        k = i*(C-1)
        xlin = [0, rmax*np.cos(th[k])]
        ylin = [0, rmax*np.sin(th[k])]
        plt.plot(xlin,ylin,color='lightgray',lw=1,zorder=-10)

    #Plot the radial grid (circles)
    nalpha = 100 #Number of points to draw the circle grids
    alpha = np.linspace(0,2*np.pi,nalpha)
    alphat = 0.7 #Angle to draw the text of the radial grid
    for k,rx in enumerate(rgv):
        #Draw one circle
        xg = rx*np.cos(alpha)
        yg = rx*np.sin(alpha)
        plt.plot(xg,yg,color='lightgray',lw=1,zorder=-10)
        #Draw the text with the value of the radius of one circle
        rgt = rx-0.15*rmax
        xgt = rgt*np.cos(alphat)
        ygt = rgt*np.sin(alphat)
        t = int(rx)
        t = FormTgrid[k].format(rgvt[k])
        plt.text(xgt,ygt,t,color='gray',ha='left',va='bottom')
    
    #Plot the label of classes
    if innerlabel:
        rt = 1.15 *rmax
    else:
        rt = 1.07 *rmax
    rts = 1.05 *rmax
    for i in range(C):
        k = i*(C-1)
        dth = (th[k+1]-th[k])
        tht = th[k]+ (C-1)/2*dth
        xt = rt*np.cos(tht)
        yt = rt*np.sin(tht)
        if rotoutlabel:
            rot = tht*180/np.pi
            if rot>90 and rot<270:
                rot = rot-180
        else:
            rot = 0
        if outerlabel:
            plt.text(xt,yt,cl[i],color='k',ha='center',va='center',rotation=rot)
        if innerlabel:
            subsecv = np.arange(0,C)
            subsecv = np.delete(subsecv,i)
            for j,ind in enumerate(subsecv):
                thts = th[k]+(j+0.5)*dth
                xts = rts*np.cos(thts)
                yts = rts*np.sin(thts)
                if rotinnlabel:
                    rot = thts*180/np.pi
                    if rot>90 and rot<270:
                        rot = rot-180
                else:
                    rot = 0
                if j % 2 == 0:
                    plt.text(xts,yts,ind,color='lightgray',
                            ha='center',va='center',rotation=rot,fontsize=3)

    #Final graphic touchs
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()


def GetConfusionSectors(em,th,beta,star,log,fill):
    '''Obtains the definition of the sectors required 
            to draw the confusion star (or confusion gear)
        These sectors include:
            - The arches (circular)
            - The connecting lines (radial)
            - The color of each sector
    Parameters
    ----------
    em : error matrix of dimension (C,C-1)
        It is a confusion matrix with redundancies (diagonal) removed
    th : Vector of dimension C·(C-1)
        Angles of the sectors in the radial plot
    beta : Vector of dimension C
        Angles where starts each actual class in the radial plot
    star: bool (optional). A flag used to indicate the type of plot
        if True it plots a confusion star
        if False it plots a confusion gear
    log: bool. A flag used to indicate the scale of the plot.
        It only applies to confusion star plots (not to confusion gear plots)
        if True it uses log scale
        if False it uses linear scale
    fill: bool.
        If True the sectors are filled

    Returns
    -------
    sectors : list containing the following 5 elements
        - arcx: a list containing C·(C-1) elements
            Each element contains a vector with the x-coordinates of an arc
        - arcy: a list containing C·(C-1) elements
            Each element contains a vector with the y-coordinates of an arc
        - conlinx: a list containing C·(C-1) elements
            Each element contains a vector with the x-coordinates of the
            radial line connecting 2 arches: the end of the corresponding arc 
            with the beginning of the next one
        - conliny: a list containing C·(C-1) elements
            Each element contains a vector with the y-coordinates of the
            radial line connecting 2 arches: the end of the corresponding arc 
            with the beginning of the next one
        - color: a list containing C·(C-1) elements
            Each element contains the color of the corresponding arc and
            radial connecting line
    '''
    C = em.shape[0]
    TabColor = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if C>len(TabColor):
        cmap = plt.get_cmap('gist_rainbow')
        TabColor = [cmap(1.*i/C) for i in range(C)]
    nth = (C-1)*C #Number of arches
    nfi = np.maximum(int(3600/nth),2) #Number of points to draw each arc

    # Compute the radii of the arches
    if star: #Confusion star
        r = em.flatten()
        if log: #Confusion star with log scale
            rmin = 0.01
            r = np.log10(np.maximum(rmin,r))-np.log10(rmin)
            rmax = 4
        else: #Confusion star with linear scale
            rmax = np.max(r)
            rmax = 4*np.ceil(rmax/4)
    else: #Confusion gear (alway linear scale)
        r = (100-em).flatten()
        rmax = 100

    #Get the arches (arc); the lines connecting arches (conlin): and its colors
    arcx = [None]*nth #x-coordinate for the arches
    arcy = [None]*nth #y-coordinate for the arches
    conlinx = [None]*nth #x-coordinate for the connecting lines
    conliny = [None]*nth #y-coordinate for the connecting lines
    color = [None]*nth #color of the arches
    for k in np.arange(nth): #For each arc
        i = int(k/(C-1))
        dth = (beta[i+1]-beta[i]) / (C-1) #Angular width for  class
        if fill:
            color[k] = TabColor[i]
        else:
            color[k] = 'w'
        fi = np.linspace(th[k],th[k]+dth,nfi)
        arcx[k] = r[k]*np.cos(fi)
        arcy[k] = r[k]*np.sin(fi)
        thlin = th[k]+dth
        if k<nth-1:
            rs = r[k+1]
        else:
            rs = r[0]
        conlinx[k] = [r[k]*np.cos(thlin), rs*np.cos(thlin)]
        conliny[k] = [r[k]*np.sin(thlin), rs*np.sin(thlin)]
    sectors = [arcx,arcy,conlinx,conliny,color]
    return sectors


def PlotConfusionSectors(sectors,edgecolor):
    '''Draw the sectors of the confusion star (or a confusion gear)
        These sectors include:
            - The arches (circular)
            - The connecting lines (radial)
            - The color of each sector
    Parameters
    ----------
    sectors : list containing the following 5 elements
        - arcx: a list containing C·(C-1) elements
            Each element contains a vector with the x-coordinates of an arc
        - arcy: a list containing C·(C-1) elements
            Each element contains a vector with the y-coordinates of an arc
        - conlinx: a list containing C·(C-1) elements
            Each element contains a vector with the x-coordinates of the
            radial line connecting 2 arches: the end of the corresponding arc 
            with the beginning of the next one
        - conliny: a list containing C·(C-1) elements
            Each element contains a vector with the y-coordinates of the
            radial line connecting 2 arches: the end of the corresponding arc 
            with the beginning of the next one
        - color: a list containing C·(C-1) elements
            Each element contains the color of the corresponding arc and
            radial connecting line
    edgecolor: color specification or None.
        Star edge color

    Returns
    -------
    PlotElem : list containing the following 3 elements
        - PlotElemArc: a list containing C·(C-1) elements
            Each element contains a matplotlib Line2D object defining the
            arc corresponding to a sector
        - PlotElemConLin: a list containing C·(C-1) elements
            Each element contains a matplotlib Line2D object defining the
            connecting line corresponding to a sector
        - PlotFilledArea: a list containing C·(C-1) elements
            Each element contains a matplotlib Patch Polygon object defining
            the filled area corresponding to a sector
    '''
    arcx,arcy,conlinx,conliny,color = sectors
    nth = len(arcx) #Number of arches
    nfi = len(arcx[0]) #Number of points to draw each arc
    ax = plt.gca()
    #Get plot elements (arches, connecting lines, areas,...)
    PlotElemArc = [None]*nth
    PlotElemConlin = [None]*nth
    PlotElemFilledArea = [None]*nth
    for k in np.arange(nth):
        #print(k)
        if edgecolor==None:
            ec = color[k]
        else:
            ec = edgecolor
        PlotElemArc[k], = plt.plot(arcx[k],arcy[k],color=ec,linewidth=0.3)
        PlotElemConlin[k], = plt.plot(conlinx[k],conliny[k],color=ec,linewidth=0.3)
        #Filled area
        v = np.zeros((nth*nfi,2)) #vertices of the polygon defining filled area
        v[k*nfi:(k+1)*nfi,0] = arcx[k]
        v[k*nfi:(k+1)*nfi,1] = arcy[k]
        PlotElemFilledArea[k] = patches.Polygon(v,color=color[k],alpha=0.1)
        ax.add_patch(PlotElemFilledArea[k])
    #Pack all plot elements
    PlotElem = []
    PlotElem.extend(PlotElemArc)
    PlotElem.extend(PlotElemConlin)
    PlotElem.extend(PlotElemFilledArea)
    return PlotElem

def cm2um(cm,unit):
    '''Convert a confusion matrix to an unit confusion matrix if required
    Parameters
    ----------
    cm : confusion matrix of dimension (C,C)
    unit : bool
        A flag used to indicate if the unit confusion matrix has to be obtained

    Returns
    -------
    cm : confusion matrix of dimension (C,C): no changes to the matrix
    OR
    um : unit confusion matrix of dimension (C,C)
    '''    
    if unit:
        C = cm.shape[0] #Number of classes
        um = np.zeros((C,C))
        um[:,:] = cm
        m = np.sum(cm,axis=1) #Number of instances per class
        for i in range(C):
            um[i,:] = um[i,:]/m[i]*100
        return um
    else:
        return cm


def um2em(um):
    '''Remove the redundancies in a confusion matrix.
        By "redundancy" we mean the elements of the diagonal
    Parameters
    ----------
    um : confusion matrix (plain or unit) of dimension (C,C)

    Returns
    -------
    em : error matrix of dimension (C,C-1)
        It is a confusion matrix with redundancies (diagonal) removed
    '''    
    C = um.shape[0] #Number of classes
    em = np.zeros((C,C-1))
    for i in range(C):
        em[i,:i] = um[i,:i]
        em[i,i:] = um[i,i+1:]
    return em

def PlotColoredGrid(cm,unit=True,xyticks='full',celltext=True):
    '''Plot a colored grid
    Parameters
    ----------
    cm: confusion matrix of dimension (C,C)
    unit: bool
        A flag used to indicate if the unit confusion matrix has to be obtained
    xyticks: string
        Indicate if the ticks and labels on X and Y axis has to be drawn
        If 'auto', ticks and labels are automatically determined
        If None, ticks and labels are ommited
        If 'full' or any other value, ticks and labels are drawn for every class
    celltext: bool (optional).
        If True the value of each cell is drawn
    '''        
    C = cm.shape[0] #Number of classes
    um = cm2um(cm,unit) #Convert to unit matrix if required
    if unit:
        formato = '{:3.0f}%'
        clim = [0,100]
    else:
        formato = '{:3d}'
        clim = None
    plt.figure()
    ax = plt.gca()
    plt.imshow(um,cmap='Spectral_r',clim=clim)
    if xyticks=='auto':
        ax.xaxis.tick_top()
    elif xyticks==None:
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    else: #opción elegida también para "full"
        plt.xticks(ticks=range(C))
        ax.xaxis.tick_top()
        plt.yticks(ticks=range(C))
    plt.xlabel('Estimated class')
    plt.ylabel('Actual class')
    ax.xaxis.set_label_position('top') 
    plt.colorbar()
    if celltext:
        for i in range(C):
            for j in range(C):
                t = formato.format(um[i,j])
                plt.text(j,i,t,ha='center',va='center')
    plt.tight_layout()

def GetSequenceConfusionMatrices(filename):
    '''Reads an Excel file containing a sequence of confusion matrices
    Parameters
    ----------
    filename : str
        The name of the file
        It has to be an excel file with the following structure:
            - The first row indicates the names of the estimated classes
            - The first column (rows 2 to C+1) indicates the names
                of the actual classes
            - Beginning in the cell (2,2) the first (C,C) confusion matrix
            - Beginning in the cell (C+2,2) the second (C,C) confusion matrix
            - The same structure is repeated for each confusion matrix 
                in the sequence
            
    Returns
    -------
    Lcm : a list of elements 
        Each element is a confusion matrix in the sequence
    '''    
    data = pd.read_excel(filename, index_col=0).values
    n,C = data.shape
    nseq = int(n/C) #Number of matrices in the sequence
    Lcm = [None]*nseq #List of confusion matrices
    for iseq in range(nseq):
        Lcm[iseq] = data[iseq*C:iseq*C+C,:]
    return Lcm


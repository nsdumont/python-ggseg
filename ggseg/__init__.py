__version__ = '0.1'


import os.path as op
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from scipy.ndimage import zoom
from skimage.transform import resize
import glob
import ggseg

def find_matching_file(data_key, wd, use_underscore=True):
    """
    Find matching file for a data key, handling case and separator variations.
    
    Parameters:
    -----------
    data_key : str
        Key from data dictionary
    wd : str
        Working directory containing SVG files
        
    Returns:
    --------
    str or None
        Full path to matching file, or None if not found
    """
    # Normalize the input key
    if use_underscore:
        normalize = lambda x: x.lower().replace('-', '_') 
    else:
        normalize = lambda x: x.lower().replace('_', '-') 
    normalized_key = normalize(data_key) 
    
    # Get all files in directory
    import glob
    all_files = glob.glob(op.join(wd, '*'))
    
    # Check each file
    for file_path in all_files:
        filename = op.basename(file_path)
        # Remove extension if present
        filename_base = op.splitext(filename)[0]
        
        # Normalize filename for comparison
        normalized_filename = normalize(filename_base)
        
        if normalized_filename == normalized_key:
            return file_path
    
    # If no match found, return None
    return None


def extract_all_values(data):
    """
    Extract all scalar values from mixed data (scalars and arrays) for colormap normalization.
    
    Parameters:
    -----------
    data : dict
        Dictionary with values that can be scalars or 2D arrays
        
    Returns:
    --------
    list
        List of all scalar values for min/max computation
    """
    all_values = []
    
    for v in data:
        if isinstance(v, np.ndarray):
            flat_values = v.flatten()
            finite_values = flat_values[np.isfinite(flat_values)]
            all_values.extend(finite_values.tolist())
        else:
            all_values.append(v)
    return all_values

def get_cmap_mixed_data(cmap, data, vminmax=[]):
    """
    Modified version of ggseg._get_cmap_ that handles mixed scalar/array data.
    
    Parameters:
    -----------
    cmap : str or matplotlib colormap
        Colormap name or object
    data : dict
        Dictionary with mixed scalar/array values
    vminmax : list, optional
        [vmin, vmax] for manual scaling, empty list for auto-scaling
        
    Returns:
    --------
    tuple
        (cmap, norm) for matplotlib plotting
    """
    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap)
    
    if vminmax == []:
        # Extract all values from mixed data
        all_values = extract_all_values(data)
        if len(all_values) == 0:
            # Fallback if no valid values found
            vmin, vmax = 0, 1
        else:
            vmin, vmax = min(all_values), max(all_values)
    else:
        vmin, vmax = vminmax
    
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def _render_data_(data, wd, cmap, norm, ax, edgecolor, use_underscore=True):
    """
    Render data in brain region patches.
    
    Parameters:
    -----------
    data : dict
        Dictionary where keys are region names and values can be:
        - scalar: single color value (original behavior)
        - 2D numpy array: image data to be plotted within the patch
        Keys are matched case-insensitively and handle underscore/hyphen variations
    wd : str
        Working directory containing SVG patch files
    cmap : matplotlib colormap
        Colormap for rendering
    norm : matplotlib normalizer
        Normalizer for color scaling
    ax : matplotlib axis
        Axis to plot on
    edgecolor : color
        Edge color for patches
    """
    
    for k, v in data.items():
        # Find matching file with flexible key matching
        fp = find_matching_file(k, wd, use_underscore)
        
        if fp is not None:
            p = open(fp).read()
            codes, verts = _svg_parse_(p)
            path = Path(verts, codes)
            
            # Check if value is an array or scalar
            if isinstance(v, np.ndarray) and v.ndim == 2:
                # Handle array data
                _render_array_in_patch(v, path, cmap, norm, ax, edgecolor)
            else:
                # Handle scalar data (original behavior)
                c = cmap(norm(v))
                ax.add_patch(patches.PathPatch(path, facecolor=c,
                                             edgecolor=edgecolor, lw=1, zorder=2))
        else:
            print(f'No matching file found for key: {k}')
            pass

def _render_array_in_patch(array_data, path, cmap, norm, ax, edgecolor):

    bounds = path.get_extents()
    x_min, y_min = bounds.x0, bounds.y0
    x_max, y_max = bounds.x1, bounds.y1
    
    patch_width = x_max - x_min
    patch_height = y_max - y_min
    
    # Determine grid resolution
    target_width = max(30, int(patch_width * 1.5))
    target_height = max(30, int(patch_height * 1.5))
    
    # Resize array
    resized_array = resize(array_data, (target_height, target_width), 
                          preserve_range=True, anti_aliasing=True)
    
    # Create coordinate grids
    x_edges = np.linspace(x_min, x_max, target_width + 1)
    y_edges = np.linspace(y_min, y_max, target_height + 1)
    
    # Create mask
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([X_centers.ravel(), Y_centers.ravel()])
    mask = path.contains_points(points).reshape(X_centers.shape)
    
    # Apply mask
    masked_data = resized_array.copy()
    masked_data[~mask] = np.nan
    
    # Plot using pcolormesh
    mesh = ax.pcolormesh(x_edges, y_edges, masked_data, 
                        cmap=cmap, norm=norm, shading='flat', alpha=0.8, zorder=2)
    
    # Add boundary
    patch_boundary = patches.PathPatch(path, facecolor='none', 
                                     edgecolor=edgecolor, lw=1, zorder=3)
    ax.add_patch(patch_boundary)


def _svg_parse_(path):
    import re
    import numpy as np
    from matplotlib.path import Path
    vertices = []
    codes = []
    
    # Clean the path string
    path_string = path.strip()
    
    # Split on commands while preserving them
    tokens = re.split(r'([MmLlHhVvCcSsQqTtAaZz])', path_string)
    tokens = [t.strip() for t in tokens if t.strip()]
    
    current_pos = np.array([0.0, 0.0])
    path_start = np.array([0.0, 0.0])
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Skip empty tokens
        if not token or token.isspace():
            i += 1
            continue
            
        # Check if this is a command
        if token in 'MmLlHhVvCcSsQqTtAaZz':
            cmd = token
            
            # Get the coordinate data following this command
            coord_data = ""
            if i + 1 < len(tokens) and tokens[i + 1] not in 'MmLlHhVvCcSsQqTtAaZz':
                coord_data = tokens[i + 1]
                i += 1  # Skip the coordinate data in next iteration
            
            # Parse coordinates
            coords = parse_coordinates(coord_data)
            
            # Process the command
            if cmd.upper() == 'M':  # Move to
                if len(coords) >= 2:
                    x, y = coords[0], coords[1]
                    
                    if cmd.islower():
                        x += current_pos[0]
                        y += current_pos[1]
                    
                    vertices.append([x, y])
                    codes.append(Path.MOVETO)
                    current_pos = np.array([x, y])
                    path_start = current_pos.copy()
                    
                    # Handle additional coordinate pairs as line-to commands
                    for j in range(2, len(coords), 2):
                        if j + 1 < len(coords):
                            x, y = coords[j], coords[j + 1]
                            if cmd.islower():
                                x += current_pos[0]
                                y += current_pos[1]
                            vertices.append([x, y])
                            codes.append(Path.LINETO)
                            current_pos = np.array([x, y])
            
            elif cmd.upper() == 'C':  # Cubic Bézier curve
                # Inkscape B-splines use cubic Bézier curves
                for j in range(0, len(coords), 6):
                    if j + 5 < len(coords):
                        cp1_x, cp1_y = coords[j], coords[j + 1]      # First control point
                        cp2_x, cp2_y = coords[j + 2], coords[j + 3]  # Second control point  
                        end_x, end_y = coords[j + 4], coords[j + 5]  # End point
                        
                        if cmd.islower():
                            cp1_x += current_pos[0]
                            cp1_y += current_pos[1]
                            cp2_x += current_pos[0]
                            cp2_y += current_pos[1]
                            end_x += current_pos[0]
                            end_y += current_pos[1]
                        
                        # Add the three points for the cubic Bézier curve
                        vertices.extend([
                            [cp1_x, cp1_y],  # First control point
                            [cp2_x, cp2_y],  # Second control point
                            [end_x, end_y]   # End point
                        ])
                        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                        current_pos = np.array([end_x, end_y])
            
            elif cmd.upper() == 'L':  # Line to
                for j in range(0, len(coords), 2):
                    if j + 1 < len(coords):
                        x, y = coords[j], coords[j + 1]
                        
                        if cmd.islower():
                            x += current_pos[0]
                            y += current_pos[1]
                        
                        vertices.append([x, y])
                        codes.append(Path.LINETO)
                        current_pos = np.array([x, y])
            
            elif cmd.upper() == 'Z':  # Close path
                if len(vertices) > 0:
                    vertices.append([path_start[0], path_start[1]])
                    codes.append(Path.CLOSEPOLY)
                    current_pos = path_start.copy()
        
        i += 1
    
    if not vertices:
        return np.array([]), np.array([]).reshape(0, 2)
    
    return np.array(codes), np.array(vertices)

def parse_coordinates(coord_string):
    """
    Parse coordinate string, handling scientific notation and various separators
    """
    import re
    if not coord_string:
        return []
    
    # Replace commas with spaces and normalize whitespace
    coord_string = re.sub(r'[,\s]+', ' ', coord_string.strip())
    
    # Handle negative numbers that might be concatenated (e.g., "10-5" -> "10 -5")
    coord_string = re.sub(r'(?<=[0-9])-', ' -', coord_string)
    
    # Split and convert to float
    try:
        coords = [float(x) for x in coord_string.split() if x]
        return coords
    except ValueError as e:
        print(f"Warning: Could not parse coordinates '{coord_string}': {e}")
        return []


def _add_colorbar_(ax, cmap, norm, ec, labelsize, ylabel):
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1%', pad=1)

    cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                           norm=norm,
                                           orientation='vertical',
                                           ticklocation='left')
    cb1.ax.tick_params(labelcolor=ec, labelsize=labelsize)
    cb1.ax.set_ylabel(ylabel, color=ec, fontsize=labelsize)



def _create_figure_(files, figsize, background, title, fontsize, edgecolor):
    import numpy as np
    import matplotlib.pyplot as plt

    codes, verts = _svg_parse_(' '.join(files))

    xmin, ymin = verts.min(axis=0) - 1
    xmax, ymax = verts.max(axis=0) + 1
    yoff = 0
    ymin += yoff
    verts = np.array([(x, y + yoff) for x, y in verts])

    fig = plt.figure(figsize=figsize, facecolor=background)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1,
                      xlim=(xmin, xmax),  # centering
                      ylim=(ymax, ymin),  # centering, upside down
                      xticks=[], yticks=[])  # no ticks
    ax.set_title(title, fontsize=fontsize, y=1.03, x=0.55, color=edgecolor)
    return ax


def _render_regions_(files, ax, facecolor, edgecolor):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    for f in files:
        codes, verts = _svg_parse_(f)
        path = Path(verts, codes)
    
        ax.add_patch(patches.PathPatch(path, facecolor=facecolor,
                                       edgecolor=edgecolor, lw=1, zorder=1))


def _get_cmap_(cmap, values, vminmax=[]):
    import matplotlib

    cmap = matplotlib.cm.get_cmap(cmap)
    if vminmax == []:
        vmin, vmax = min(values), max(values)
    else:
        vmin, vmax = vminmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def plot_dk(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
             figsize=(15, 15), bordercolor='w', vminmax=[], title='',
             fontsize=15, whole_reg = ['lateral_left', 'medial_left', 'lateral_right',
                 'medial_right']):
    """Plot cortical ROI data based on a Desikan-Killiany (DK) parcellation.

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the cortical Desikan-Killiany atlas. The
            full list of applicable regions can be found in the folder
            ggseg/data/dk.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'dk')

    # A figure is created by the joint dimensions of the whole-brain outlines
    
    files = [open(op.join(wd, e)).read() for e in whole_reg]
    ax = _create_figure_(files, figsize, background, title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = get_cmap_mixed_data(cmap, data.values(), vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor, use_underscore=True)

    # DKT regions with no provided values are rendered in gray
    data_regions = list(data.keys())
    dkt_regions = [op.splitext(op.basename(e))[0] for e in reg]
    NA = set(dkt_regions).difference(data_regions).difference(whole_reg)
    files = [open(op.join(wd, e)).read() for e in NA]
    _render_regions_(files, ax, 'lightgray', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()

def plot_hippo(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
             figsize=(15, 15), bordercolor='w', vminmax=[], title='',
             fontsize=15):
    """

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the hippocampus + EC. The
            full list of applicable regions can be found in the folder
            ggseg/data/hippo.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'hippo')

    # A figure is created by the joint dimensions of the whole-brain outlines
    whole_reg = ['formation']
    files = [open(op.join(wd, e)).read() for e in whole_reg]
    ax = _create_figure_(files, figsize, background, title, fontsize, edgecolor)
    _render_regions_(files, ax, bordercolor, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg if whole_reg[0] not in e]
    _render_regions_(files, ax, 'lightgrey', edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = get_cmap_mixed_data(cmap, data.values(), vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor, use_underscore=True)


    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()

def plot_aseg(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
              figsize=(15, 5), bordercolor='w', vminmax=[],
              title='', fontsize=15, whole_reg = ['Coronal', 'Sagittal']):
    """Plot subcortical ROI data based on the FreeSurfer `aseg` atlas

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the FreeSurfer `aseg` atlas. The full list
            of applicable regions can be found in the folder ggseg/data/aseg.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'aseg')
    reg = [op.basename(e) for e in glob(op.join(wd, '*'))]

    # Select data from known regions (prevents colorbar from going wild)
    known_values = []
    for k, v in data.items():
        if k in reg:
            known_values.append(v)

    files = [open(op.join(wd, e)).read() for e in whole_reg]

    # A figure is created by the joint dimensions of the whole-brain outlines
    ax = _create_figure_(files, figsize, background,  title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = get_cmap_mixed_data(cmap, known_values, vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor,use_underscore=False)

    # The following regions are ignored/displayed in gray
    # NA = ['Cerebellum-Cortex', 'Cerebellum-White-Matter', 'Brain-Stem']
    # files = [open(op.join(wd, e)).read() for e in NA]
    # _render_regions_(files, ax, '#111111', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()

def plot_jhu(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
             figsize=(17, 5), bordercolor='w', vminmax=[], title='',
             fontsize=15):
    """Plot WM ROI data based on the Johns Hopkins University (JHU) white
    matter atlas.

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the Johns Hopkins University white Matter
            atlas. The full list of applicable regions can be found in the
            folder ggseg/data/jhu.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """

    import matplotlib.pyplot as plt
    import ggseg
    import os.path as op
    from glob import glob

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'jhu')

    # A figure is created by the joint dimensions of the whole-brain outlines
    whole_reg = ['NA']
    files = [open(op.join(wd, e)).read() for e in whole_reg]
    ax = _create_figure_(files, figsize, background, title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = get_cmap_mixed_data(cmap, data.values(), vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor)

    # JHU regions with no provided values are rendered in gray
    NA = ['CSF']
    files = [open(op.join(wd, e)).read() for e in NA]
    _render_regions_(files, ax, 'gray', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()


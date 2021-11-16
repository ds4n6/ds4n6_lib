###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4gui

###############################################################################
# IMPORTS
###############################################################################

# DEV  IMPORTS ----------------------------------------------------------------

# python IMPORTS --------------------------------------------------------------
import os
import sys
import io
import datetime
import functools
import re
import inspect
import webbrowser
import base64

# DS IMPORTS ------------------------------------------------------------------
import pandas as pd
from pandas.api.types import is_string_dtype, is_categorical_dtype
import ipywidgets as widgets
from ipywidgets import Layout
from IPython import get_ipython

from IPython.display import display, Markdown, HTML, Javascript
from traitlets import traitlets

import qgrid
from ipyaggrid import Grid

# DS4N6 IMPORTS ---------------------------------------------------------------

import ds4n6_lib.d4          as d4
import ds4n6_lib.common      as d4com
import ds4n6_lib.gui         as d4gui
import ds4n6_lib.utils       as d4utl

# Tool / Artifact specific libs
import ds4n6_lib.autoruns    as d4_autoruns
import ds4n6_lib.evtx        as d4_evtx
import ds4n6_lib.fstl        as d4_fstl
import ds4n6_lib.kansa       as d4_kansa
import ds4n6_lib.kape        as d4_kape
import ds4n6_lib.macrobber   as d4_macrobber
import ds4n6_lib.mactime     as d4_mactime
import ds4n6_lib.plaso       as d4_plaso
import ds4n6_lib.volatility  as d4_volatility
import ds4n6_lib.tshark      as d4_tshark

###############################################################################
# FUNCTIONS
###############################################################################
curf = ""

def file_select(rootpath="/", readfunc="", compname=None, f2read="", nbsave_prefix="", **kwargs):
    file_folder_select(rootpath, readfunc, compname, f2read, nbsave_prefix, **kwargs)

def folder_select(rootpath="/", readfunc="", compname=None, f2read="", nbsave_prefix="", **kwargs):
    file_folder_select(rootpath, readfunc, compname, f2read, nbsave_prefix, **kwargs)

def file_folder_select(rootpath="/", readfunc="", compname=None, f2read="", nbsave_prefix="", **kwargs):
    """ Function to select a file o folder to read
    """
    # **kwargs are passed through to readfunc()
    def read_selected_file(b, compname=None, f2read=f2read, nbsave_prefix="", **kwargs):
        global curf

        notebook_file = kwargs.get('notebook_file', "")

        if f2read == "":
            with output:
                curf   = d4utl.path
                infile = d4utl.path

                print("Selected file/folder:")
                print("    "+curf)
                print("")
                
                print("- Reading file(s):")
                d4.out = {}
                d4.out = readfunc(infile, **kwargs)
                print("- Reading Done.\n")
                print("- Recover the object read by executing:  myvar = d4.out (e.g. mydf = d4.out)")
                print("")

                savepathtonb = True
                if savepathtonb :
                    print("- Saving f2read path to notebook cell")
                    if notebook_file == "":
                        print("  + notebook_file not defined. Cannot save.")
                    else:
                        hits = d4utl.nbgrep(notebook_file, nbsave_prefix+'_f2read = "')
                        if not hits:
                            cell_a = "# Automatically created - DO NOT EDIT OR REMOVE unless you want to change the file to read (in that case, remove this cell)\n"
                            cell_b = "# "+str(datetime.datetime.now())+"\n"
                            cell_c = '# '+nbsave_prefix+'_f2read = "'+curf+'"'
                            cell   = cell_a + cell_b + cell_c
                            get_ipython().set_next_input(cell)

        else:
            infile = f2read

            print("- Reading file(s):")
            d4.out = {}
            if compname is not None:
                d4.out = readfunc(infile, compname=compname, **kwargs)
            else:
                d4.out = readfunc(infile, **kwargs) 

            print("")

            if d4.out is None:
                print("ERROR: No valid data read. Aborting.")
                return
            else:
                if not len(d4.out) == 0:
                    print("- Reading Done.\n")
                    print("- Recover the object read by executing:  myvar = d4.out (e.g. mydf = d4.out)")
                    print("")


    if f2read != "":
        dummy = False
        read_selected_file(dummy, compname=compname, f2read=f2read, nbsave_prefix=nbsave_prefix, **kwargs)
    else:
        if rootpath == "":
            rootpath = "/"

        display(Markdown("Select "+nbsave_prefix+" file:"))

        if not os.path.exists(rootpath):
            rootpath = "/"
        
        f = d4utl.PathSelector(rootpath)
        display(f.accord)

        button = widgets.Button(description="SELECT & READ")
        output = widgets.Output()
        output.clear_output()
        display(button, output)

        button.on_click(functools.partial(read_selected_file, compname=compname, nbsave_prefix=nbsave_prefix, **kwargs))
        

def xread(**kwargs):
    """ User interface for read data from different type of files
    Syntax: read_data_gui(tool="<tool>" [, rootpath="<rootpath>", notebook_file="<nbfile>", hostname="<hostname>", prefix="<prefix>", ext="<ext>")
    Args:
        tool (str) : tool used to generate files to readd
        rootpath (str) : path of the file/folder with files to read
        notebook_file (str) : path of notebook that is calling the function for check if the data is already readed
        hostname (str) : Hostname of the data to read
        prefix (str) :
        ext (str) :
    """
    def syntax():
        print('Syntax: xread(tool="<tool>" [, rootpath="<rootpath>", notebook_file="<nbfile>", hostname="<hostname>", prefix="<prefix>", ext="<ext>")')
        return

    nargs = len(kwargs)

    if nargs == 0:
        syntax()
        return

    tool          = kwargs.get('tool',          "")
    rootpath      = kwargs.get('rootpath',      "/")
    notebook_file = kwargs.get('notebook_file', "")
    hostname      = kwargs.get('hostname',      "")

    nbsave_prefix = hostname+'_'+tool+'d'

    if notebook_file != "":
        pattern = nbsave_prefix+'_f2read = "/.*$'

        print("- Searching notebook for saved input file / folder ("+nbsave_prefix+")")
        hits = d4utl.nbgrep(notebook_file, pattern)
        if hits:
            f2read = hits[0].split(" = ")[1].strip('"').strip('# ')
            print("  + Found: "+f2read)
        else:
            f2read = ""
            print("  + Not Found")
        print("")
    else:
        f2read = ""

    func2exec = 'd4_'+tool+'.read_data'

    d4gui.folder_select(rootpath, functools.partial(eval(func2exec), **kwargs), f2read=f2read, nbsave_prefix=nbsave_prefix, notebook_file=notebook_file)
        
        
        

# dfmenu() & friends ##########################################################
def xmenu(obj, engine="default"):
    """ User interface to explore forensics dataframes and dictionaries of dataframes

        Args: 
            obj (pandas.DataFrame or dict of pandas.DataFrame): data to exploer
            engine (str) : type of grid to show the data, options: default|qgrid|aggrid
                (default is "default")
                
        Returns: 
            dfs_explorer: instance that display the UI
    """
    if len(obj) == 0:
        print("- Empty object ({})".format(type(obj)))
        return

    objtype = d4com.data_identify(obj)
    column_mode = False
    if re.search("^pandas_dataframe-", objtype):
        dfs = dict({" + DataFrame": obj})
        for col in obj.columns:
            dfs[" -- "+col] = obj[col].to_frame()
        column_mode = True
    elif re.search("^dict-", objtype) and len(obj) > 0:
        dfs = obj
    else:
        print("- Unsupported object ({})".format(type(obj)))
        return

    explorer = dfs_explorer(dfs, column_mode=column_mode)
    explorer.display()
    return explorer 


# use:
#
#   grid = d4gui.aggrid(df)
#   grid
#
# get selected rows:
#    grid.grid_data_out.get('rows')
def aggrid(df, page_size=10):
    """ Function description

        Args: 
            df (pandas.DataFrame): data to show
            page_size (int): number of rows per page
                (default is 10)

        Returns: 
            ipyaggrid.Grid: object that show the data using ag-grid JS framework
        
    """
    timestamp_cellrenderer='''
    function (params){
        return params.value;
    }
    '''
    column_defs = [{'headerName': df.index.name, 'field': df.index.name, 'checkboxSelection': True, 'cellRenderer': timestamp_cellrenderer if df.index.name == "timestamp" else None}]
    column_defs +=  [{'headerName': c, 'field': c, 'cellRenderer': timestamp_cellrenderer if c == "timestamp" else None} for c in df.columns]
    gridOptions = {
        'columnDefs': column_defs,
        'defaultColDef': {'sortable': True, 'filter': True, 'resizable': True, 'floatingFilter': True},
        'animateRows': True,
        'enableRangeSelection': True,
        'enableColResize': True,
        'rowSelection': 'multiple',
        'enableFilter': True,
        'enableSorting': True,
        'pagination':True,
        'paginationPageSize': page_size
    }
    grid = Grid(
        grid_options=gridOptions, 
        grid_data=df,
        quick_filter=True,
        columns_fit="auto",
        export_csv=True,
        export_excel=True,
        export_mode='auto',
        export_to_df=True,
        keep_multiindex=False,
        theme="ag-theme-balham",
        index=True 
    )
    return grid

class dfs_explorer():
    """ A class used to display a dataframes explorer

    Atributes:
        current_df (pandas.DataFrame) :
        current_df_name (str) :
        simple (bool) : 
    
    Methods: 
        update_options()
        update_grid()
        reload_df(name)
        display()
        export_grid_to_df()
    """
    current_df=None
    current_df_name=""
    simple=True
    

    def __init__(self, dfs, grid_type="default", column_mode=False):
        self.dfs = {}
        for key in dfs.keys():
            if not dfs[key].empty:
                self.dfs[key] = dfs[key]
        # self.dfs = dfs
        if grid_type in ['default', 'qgrid', 'aggrid']: 
            self.grid_type = grid_type
        box_layout = Layout(overflow='auto', width='99%')
        self.dfout = widgets.Output(layout=box_layout)     
        self.dfout.clear_output()
        self.footer = widgets.Output(layout=box_layout)

        def simple_output_selector_eventhandler(change):
            self.simple = change.new
            self.reload_df(self.current_df_name)

        def select_df_eventhandler(change):
            self.reload_df(change.new)
        
        def select_grid_eventhandler(change):
            self.grid_type = change.new
            self.reload_df(self.current_df_name)
        
        def mixs(num): 
            try: 
                ele = int(num) 
                return (0, ele, '') 
            except ValueError: 
                return (1, num, '') 

        dropdown_description = 'Select DataFrame'
        self.column_mode = column_mode
        if column_mode:
            dropdown_description += " or Column"
        options=[dropdown_description]
        dfs_keys = list(self.dfs.keys())
        dfs_keys.sort(key = mixs)
        options.extend(dfs_keys)
        self.dropdown = widgets.Dropdown(layout={'width': '550px'}, options=options, description=dropdown_description, disabled=False, style={'description_width': 'initial'})
        self.dropdown.observe(select_df_eventhandler, names='value')

        grid_options=['aggrid', 'qgrid', 'default']
        self.grid_selection = widgets.Dropdown(layout={'width': '250px'}, options=grid_options, description='Select grid', disabled=False, value=self.grid_type)
        self.grid_selection.observe(select_grid_eventhandler, names='value')

        self.max_rows = 20
        max_rows_options=['20', '50', '100', '200', '500', '1000']
        def select_max_rows_eventhandler(change):
            self.max_rows = int(change.new)
            self.reload_df(self.current_df_name)
        self.max_rows_selection = widgets.Dropdown(layout={'width': '250px'}, options=max_rows_options, description='Select max_rows', disabled=False, value=str(self.max_rows), style={'description_width': 'initial'})
        self.max_rows_selection.observe(select_max_rows_eventhandler, names='value')

        #if re.search("^pandas_dataframe-evtx_file_df", objtype) or re.search("^pandas_dataframe-evtx_file_raw_df", objtype):
        self.simple_check = widgets.Checkbox(self.simple, description='Simple Output')
        self.simple_check.observe(simple_output_selector_eventhandler, names='value')
 
        self.consolidate_cols = True
        self.consolidate_cols_check = widgets.Checkbox(self.consolidate_cols, description='Consolidate cols.')    
        def consolidate_cols_check_eventhandler(change):
            self.consolidate_cols = change.new
            self.reload_df(self.current_df_name)
        self.consolidate_cols_check.observe(consolidate_cols_check_eventhandler, names='value')
        
        self.collapse_constant_cols = True
        self.collapse_constant_cols_check = widgets.Checkbox(self.collapse_constant_cols, description='Collapse constant cols.')
        def collapse_constant_cols_check_eventhandler(change):
            self.collapse_constant_cols = change.new
            self.reload_df(self.current_df_name)
        self.collapse_constant_cols_check.observe(collapse_constant_cols_check_eventhandler, names='value')
        
        self.hide_cols = True
        self.hide_cols_check = widgets.Checkbox(self.hide_cols, description='Hide cols.')
        def hide_cols_check_eventhandler(change):
            self.hide_cols = change.new
            self.reload_df(self.current_df_name)
        self.hide_cols_check.observe(hide_cols_check_eventhandler, names='value')
        
        
        self.apply_filters = True
        self.apply_filters_check = widgets.Checkbox(self.apply_filters, description='Apply Filters')
        def apply_filters_check_eventhandler(change):
            self.apply_filters = change.new
            self.reload_df(self.current_df_name)
        self.apply_filters_check.observe(apply_filters_check_eventhandler, names='value')
        
        self.wrap_cols = True
        self.wrap_cols_check = widgets.Checkbox(self.wrap_cols, description='Wrap cols.')
        def wrap_cols_check_eventhandler(change):
            self.wrap_cols = change.new
            self.reload_df(self.current_df_name)
        self.wrap_cols_check.observe(wrap_cols_check_eventhandler, names='value')
        
        
        self.out = True        
        self.out_df = False        
        
        self.beautify_cols = False
        self.beautify_cols_check = widgets.Checkbox(self.beautify_cols, description='Beautify cols.')
        def beautify_cols_check_eventhandler(change):
            self.beautify_cols = change.new
            self.reload_df(self.current_df_name)
        self.beautify_cols_check.observe(beautify_cols_check_eventhandler, names='value')
        
        self.ret = True        
        self.ret_out = True
        self.ret_out_check = widgets.Checkbox(self.ret_out, description='Ret out.')
        def ret_out_check_eventhandler(change):
            self.ret_out = change.new
            self.reload_df(self.current_df_name)
        self.ret_out_check.observe(ret_out_check_eventhandler, names='value')
        
        self.menu = widgets.Box([
            self.dropdown,
            self.grid_selection,
            self.simple_check,
            self.max_rows_selection
        ])

        self.options_dict = {
            'consolidate_cols': self.consolidate_cols_check,
            'collapse_constant_cols': self.collapse_constant_cols_check,
            'hide_cols': self.hide_cols_check,
            'apply_filters': self.apply_filters_check,    
            'wrap_cols': self.wrap_cols_check,    
            'beautify_cols': self.beautify_cols_check,
            'ret_out': self.ret_out_check,
        }
        
        self.select_df_button = widgets.Button(description="Export Grid to DataFrame", icon="download", layout={'width': '250px'})
        def select_df_button_eventhandler(change):
            self.export_grid_to_df()
        self.select_df_button.on_click(select_df_button_eventhandler)
        
        # Loading DF section
        loading_label = widgets.HTML("<div style='color: orange; font-weight: bold;'>Data Loading. Please Wait…</div>")
        self.loading = widgets.HBox([loading_label])
        self.loading.layout.visibility = 'hidden'
        self.loading.layout.height = '0px'

        # Selected DF Section
        self.selected_df_label = widgets.Label(value="Selected dataframe: ")
        self.selected_df = widgets.HBox([self.selected_df_label])


    def update_options(self):
        if not self.current_df_name:
            return
        source_type = d4com.whatis(self.dfs[self.current_df_name]) 
        if "-" in source_type:
            source_type = source_type.split("-")[1]
        if "_" in source_type:
            source_type = source_type.split("_")[0]
        srcoptions = d4com.get_source_options(source_type)
        options_items = [self.options_dict['collapse_constant_cols'], self.options_dict['hide_cols'], self.options_dict['ret_out']]
        for option in srcoptions:
            options_items.append(self.options_dict[option])
        self.simple_options = widgets.GridBox(options_items, layout=widgets.Layout(grid_template_columns="repeat(5, 200px)"))

    def update_grid(self):
        if not self.current_df_name :
            return
        if self.simple and not self.ret_out:
            self.grid = None
        elif self.grid_type == 'aggrid':
            try:
                for col in self.current_df.columns:
                    if self.current_df[col].dtype == 'Int64':
                        self.current_df[col] = self.current_df[col].astype("int64")
                self.grid = aggrid(self.current_df, page_size=self.max_rows)
            except:
                display(Markdown("<span style=\"color:red\">*DF unavailable to show in aggrid, please use default or qgrid*</span>"))
                return
        elif self.grid_type == 'qgrid':
            self.current_df.insert(0,'Selected', False)
            # qgrid filter issue
            for col in self.current_df.columns:
                if is_string_dtype(self.current_df[col]):
                    self.current_df[col] = self.current_df[col].astype("object")
                if is_categorical_dtype(self.current_df[col]) and len(self.current_df[col].cat.categories) == 0:
                    self.current_df[col].cat.add_categories("", inplace=True)
            self.grid = qgrid.show_grid(self.current_df, show_toolbar=True, grid_options={'forceFitColumns':False, 'maxVisibleRows': self.max_rows})
        else:
            pd.set_option("display.max_rows", self.max_rows)
            pd.set_option("display.min_rows", self.max_rows)
            pd.set_option('display.latex.escape', False)
            self.grid = self.current_df
        with self.dfout:
            if not self.simple:
                display(Markdown("  * No. Entries: {}".format(len(self.current_df))))
            if self.grid is not None:
                display(self.grid)
                display(Markdown("  * Access your selected DataFrame via the d4.out variable"))
                display(self.select_df_button)
        

    def reload_df(self, name):
        self.loading.layout.visibility = 'visible'
        self.loading.layout.height = '30px'
        self.selected_df_label.value = "Selected DataFrame: {}".format(name)
        self.dfout.clear_output()
        self.footer.clear_output()
        if str(name).startswith("Select DataFrame") or name == "":
            self.current_df_name = ""
            self.current_df = None
            return
        with self.dfout:
            self.current_df_name=name
            self.update_options()
            if self.column_mode and self.simple and len(self.dfs[name].columns) == 1:
                self.current_df = self.dfs[name]
                self.current_df[self.current_df.columns[0]].simple()
            else:
                display(Markdown("**Simple Options:**"))
                display(self.simple_options)
                if self.simple:            
                    self.current_df=self.dfs[name].spl(
                        consolidate_cols=self.consolidate_cols,
                        collapse_constant_cols=self.collapse_constant_cols,
                        hide_cols=self.hide_cols,
                        apply_filters=self.apply_filters,
                        out=self.out,
                        out_df=self.out_df,
                        beautify_cols=self.beautify_cols,
                        ret=self.ret,
                        ret_out=self.ret_out
                    )
                else:
                    self.current_df = self.dfs[name]             
                self.update_grid()
        self.loading.layout.visibility = 'hidden'
        self.loading.layout.height = '0px'
            
    def display(self):   
        display(Markdown("**DataFrame visualization menu:**"))
        display(self.menu)
        display(self.selected_df)
        display(self.loading)
        display(self.dfout)
        display(self.footer)

    def export_grid_to_df(self):
        self.footer.clear_output()
        if not self.current_df_name :
            with self.footer:
                display(Markdown("_Nothing to export_"))
            return
        if self.grid_type == 'aggrid':
            d4.out = self.grid.grid_data_out.get('rows')
            message = "d4.out variable updated with selected rows from '{}' dataframe".format(self.current_df_name)
            if d4.out is None or d4.out.size == 0:
                d4.out = self.grid.grid_data_out.get("grid")
                message = "d4.out variable updated with '{}' dataframe".format(self.current_df_name)
        elif self.grid_type == 'qgrid':
            df_changed = self.grid.get_changed_df()
            df_selected = df_changed[df_changed['Selected']]
            if df_selected.empty:
                d4.out = df_changed.drop('Selected', axis=1)
                message = "d4.out variable updated with grid content from '{}' dataframe".format(self.current_df_name)
            else:
                d4.out = df_selected.drop('Selected', axis=1)
                message = "d4.out variable updated with selected rows from '{}' dataframe".format(self.current_df_name)
        else:
            d4.out = self.current_df
            message = "d4.out variable updated with '{}' dataframe".format(self.current_df_name)
        with self.footer:
            display(Markdown("_{}_".format(message)))
            display(d4.out.info())


def create_pd_int_option(name):
    field = widgets.IntText(
        value=pd.get_option(name),
        description='{}: '.format(name),
        disabled=False,
        style={'description_width': 'initial'}
    )
    def field_change_eventhandler(change):
        pd.set_option(name, change.new)

    field.observe(field_change_eventhandler, names='value')
    return field

def create_pd_boolean_option(name):
    field = widgets.Checkbox(
        indent=False,
        value=pd.get_option(name),
        description=name,
        disabled=False,
        style={'description_width': 'initial'}
    )
    def field_change_eventhandler(change):
        pd.set_option(name, change.new)

    field.observe(field_change_eventhandler, names='value')
    return field

def create_pd_text_option(name, options):
    field = widgets.Dropdown(
        options=options,
        value=pd.get_option(name),
        description='{}: '.format(name),
        disabled=False,
        style={'description_width': 'initial'}
    )
    def field_change_eventhandler(change):
        pd.set_option(name, change.new)

    field.observe(field_change_eventhandler, names='value')
    return field

def xdisplay():
    """ User interface to change Pandas display options 
    
    Display options that you can change:
    - display.max_rows
    - display.min_rows
    - display.max_columns
    - display.colheader_justify
    - display_expand_frame_repr
    """
    display_max_rows_field = create_pd_int_option('display.max_rows')
    display_min_rows_field = create_pd_int_option('display.min_rows')
    display_max_columns_field = create_pd_int_option('display.max_columns')
    display_colheader_justify_field = create_pd_text_option('display.colheader_justify', ['right', 'left'])
    display_expand_frame_repr_field = create_pd_boolean_option('display.expand_frame_repr')
    display(Markdown('#### Pandas options'))
    display(display_max_rows_field)
    display(display_min_rows_field)
    display(display_max_columns_field)
    display(display_colheader_justify_field)
    display(display_expand_frame_repr_field)

def xanalysis(dfs):
    """ User Interface to show analysis from forensics dataframes

        Args: 
            dfs (dict of pandas.DataFrame): data for analysis
    """
    # get analysis options
    def get_analysis_options(dfs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        d4com.anl(dfs)
        anl_output = new_stdout.getvalue()
        sys.stdout = old_stdout
        lines = anl_output.split("\n")
        options = [("Select analysis type", " ")]
        for line in lines:
            if line.startswith("-") and not line.startswith("- No analysis modules"):
                sep = line.find(":")
                value = line[2:sep]
                description = line[sep+1:].split("(")[0].strip()
                options.append((description, value))
        return options
    anl_result = None
    object_selected = None
    anl_src = dfs
    dfs_src = dfs
    box_layout = Layout(overflow='auto', width='99%')
    anl_out = widgets.Output(layout=box_layout) 
    obj_type = d4com.data_identify(dfs)
    object_options = []
    dataframe_options = ['Select one...']
    def df_type(df):
        return d4com.data_identify(df).split("-")[1]

    if obj_type.startswith("dict"):
        object_options.append('Select one...')
        object_options.append('Dictionary')
        dataframe_options += tuple(dfs.keys())
        type_options = ['Select one...']
        type_options += tuple(set([df_type(dfs[df_id]) for df_id in dfs]))
        is_df = False
    else:
        type_options = [df_type(dfs), ]
        is_df = True
    object_options.append('DataFrame')
    

    object_selection = widgets.Dropdown(layout={'width': '250px'}, options=object_options, description='Analysis object: ', disabled=is_df, style={'description_width': 'initial'})
    def object_selection_eventhandler(change):
        nonlocal object_selected
        nonlocal type_selection
        nonlocal dataframe_selection
        nonlocal second_options
        nonlocal is_df
        object_selected=change.new
        if change.new == "Dictionary":
            type_selection.layout.visibility='visible'
            type_selection.value="Select one..."
            dataframe_selection.layout.visibility='hidden'
            second_options.layout.visibility='hidden'   
        elif change.new == "DataFrame":
            type_selection.layout.visibility='visible'
            type_selection.value="Select one..."
            dataframe_selection.layout.visibility='visible'
            dataframe_selection.value="Select one..."
            if not is_df:
                second_options.layout.visibility='hidden'   
        else:
            type_selection.layout.visibility='hidden'
            dataframe_selection.layout.visibility='hidden'
            second_options.layout.visibility='hidden'   
    object_selection.observe(object_selection_eventhandler, names='value')

    type_selection = widgets.Dropdown(layout={'width': '250px'}, options=type_options, description='Analysis type: ', disabled=is_df, style={'description_width': 'initial'})
    if is_df:
        type_selection.layout.visibility='visible'
    else:
        type_selection.layout.visibility='hidden'
    def type_selection_eventhandler(change):
        nonlocal object_selected
        nonlocal dataframe_selection
        nonlocal analysis_selection
        nonlocal second_options
        nonlocal anl_src
        nonlocal anl_out
        if change.new != "Select one...":
            anl_src = d4utl.extract_dfs_from_type("pandas_dataframe-"+change.new, dfs)
            if object_selected == "Dictionary":
                analysis_selection.options = get_analysis_options(anl_src)
                if len(analysis_selection.options) == 1:
                    anl_out.clear_output()
                    with anl_out:
                        display(Markdown("**No analysis available**"))
                else:
                    second_options.layout.visibility='visible'   
                    anl_out.clear_output()
            elif object_selected == "DataFrame":
                dataframe_selection.options = ['Select one...', ]
                dataframe_selection.options += tuple(anl_src.keys())
                   
    type_selection.observe(type_selection_eventhandler, names='value')

    
    dataframe_selection = widgets.Dropdown(layout={'width': '450px'}, options=dataframe_options, description='DF to analyze: ', disabled=is_df, style={'description_width': 'initial'})
    dataframe_selection.layout.visibility='hidden'
    def dataframe_selection_eventhandler(change):
        nonlocal anl_src
        nonlocal dfs_src
        nonlocal second_options
        nonlocal analysis_selection
        nonlocal anl_out
        if change.new != "Select one...":
            anl_src = dfs_src[change.new]
            analysis_selection.options = get_analysis_options(anl_src) 
            if len(analysis_selection.options) == 1:
                anl_out.clear_output()
                with anl_out:
                    display(Markdown("**No analysis available**"))
            else:
                second_options.layout.visibility='visible'  
                anl_out.clear_output()
        else: 
            second_options.layout.visibility='hidden'  
    dataframe_selection.observe(dataframe_selection_eventhandler, names='value')

    analysis_selection = widgets.Dropdown(layout={'width': '550px'}, options=[], description='Available analysis types: ', disabled=False, style={'description_width': 'initial'})
    def select_analysis_eventhandler(change):
        nonlocal anl_result 
        nonlocal anl_src
        nonlocal anl_out
        anl_out.clear_output()
        anl_result = None
        if change.new !=" ":
            with anl_out:
                anl_result = d4com.anl(anl_src,change.new)
                display(anl_result)

    analysis_selection.observe(select_analysis_eventhandler, names='value')
    
    export_result_button = widgets.Button(description="Export Result to d4.out", icon="download", layout={'width': '250px'}, disabled=False)
    def export_result_button_eventhandler(change):
        nonlocal anl_result
        nonlocal anl_out
        if anl_result is not None:
            d4.out = anl_result
            with anl_out:
                display(Markdown("_d4.out variable updated with analisys result_"))
    export_result_button.on_click(export_result_button_eventhandler)
    
    first_options = widgets.Box([object_selection, type_selection, dataframe_selection])
    second_options = widgets.Box([analysis_selection, export_result_button])
    if is_df:
        analysis_selection.options = get_analysis_options(anl_src)
        if len(analysis_selection.options) == 1:
            anl_out.clear_output()
            with anl_out:
                display(Markdown("**No analysis available**"))
        else:
            second_options.layout.visibility='visible'
    else:
        second_options.layout.visibility='hidden'
    # anl_out.clear_output()
    display(Markdown("**Analisys explorer:**"))
    display(first_options)
    display(second_options)
    display(anl_out)

# gui short alias functions =============================================================
def xmn(*args, **kwargs):
    """ Alias for xmenu()
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return xmenu(*args, **kwargs) 

def xdp(*args, **kwargs):
    """ Alias for xdisplay()
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return xdisplay(*args, **kwargs) 

def rdg(*args, **kwargs):
    """ Alias for read_data_gui()
    """
    if d4.debug >= 2:
        print("DEBUG: [DBG"+str(d4.debug)+"] ["+str(os.path.basename(__file__))+"] ["+str(inspect.currentframe().f_code.co_name)+"()]")

    return xread(*args, **kwargs) 

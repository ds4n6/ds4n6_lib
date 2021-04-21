###############################################################################
# INFO
###############################################################################
# Recommended "import as": d4

###############################################################################
# VARIABLES
###############################################################################
# Debug Level (1: min - 5:max) ------------------------------------------------
# 1: TBD
# 2: Executed functions
# 3: Low    detail on executed functions
# 4: Medium detail on executed functions
# 5: High   detail on executed functions
debug = 0

# Other -----------------------------------------------------------------------
out = None
ipregex="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"

###############################################################################
# DECLARE VARS
# not_well-formed
main_nwf=[
     {'find':'<\x04Data', 'replace':'<Data'},
     {'find':'</\x04Data', 'replace':'</Data'},
     {'find':'\u0000', 'replace':'\\u0000'},
     {'find':'\u0001', 'replace':'\\u0001'},
     {'find':'\u0002', 'replace':'\\u0002'},
     {'find':'\u0003', 'replace':'\\u0003'},
     {'find':'\u0004', 'replace':'\\u0004'},
     {'find':'\u0005', 'replace':'\\u0005'},
     {'find':'\u0006', 'replace':'\\u0006'},
     {'find':'\u0007', 'replace':'\\u0007'},
     {'find':'\u0008', 'replace':'\\u0008'},
     {'find':'&', 'replace':'&amp;'},
     {'find':'< Name', 'replace':'<Data Name'},
     {'find':'</>', 'replace':'</Data>'},
     {'find':'    Data ', 'replace':'   <Data>'},
     {'find':' <([a-zA-Z0-9_-]*)> ', 'replace':' \\1 ', 'type':'re'},
     {'find':'::<([a-zA-Z0-9_-]*)>::', 'replace':'::\\1::', 'type':'re'},
    ]

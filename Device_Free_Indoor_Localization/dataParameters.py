######################################## Data Variables ##########################################################################
""" ##############################################################################################################################

@author: Felix Kosterhon

The data parameters are given in this class

For each measurement the following parameters are necessary:
- directory
- filesPerTarget (how many measurements per position)
- coordination information -> Start Position, End Position, Receiver ID, Transmitter ID, {} => excluded indices

""" ####################################### directory ############################################################################# #

directory = "/Path/To/sampleData"

# ----------------------------- directory of the baseline  ------------------------------------------------------------------------ #

basedirectory_ExampleRoom_2507 = directory+"/ExampleRoom_2507/Baseline/"

# ------------------------  coax directory ---------------------------------------------------------------------------------------- # 

coaxdirectory = directory+"/Coax/"

########################### In general rooms ########################################################################################

# --------------------------------- Office ---------------------------------------------------------------------------------------- #
						
directory_ExampleRoom_2507 = directory+"/ExampleRoom_2507/"

# ########################################### config ############################################################################## #

# Coordination Info:
# - Start position
# - End position
# - Receiver id
# - Transmitter id
# - {} => excluded indices

# Baseline Info -> Placeholder and can be used for all Baselines
baselineInfo = [1, 1, -7, -5, {}]

# CoaxInfo -> Placeholder and can be used for all Coax-Measurements
coaxInfo = [1, 4, -7, -5, {}]

# ------------------------------------------ Room experiments -------------------------------------------------------------------- #

# Example for a grid with 12 reference positions
coordInfo_ExampleRoom_2507 = [1, 12, -7, -5, {}]

# #################################### filesPerTarget ############################################################################ #

# For storage reasons, the provided examples are only a few
# To get reasonable performance, bigger data sets should be used.

filesCoaxPerAntenna = 10

# Number of measurements per baseline
filesBaseline_ExampleRoom_2507 = 20

# Number of measurements per position
filesPerTarget_ExampleRoom_2507 = 20

##################################################################################################################################
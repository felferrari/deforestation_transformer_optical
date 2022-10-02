
#train with year_0 and year_1 images ans test with year_1 and year_2 images
YEAR_0 = 2017
YEAR_1 = 2018
YEAR_2 = 2019

PATCH_SIZE = 128
PATCH_OVERLAP = 0.7
TEST_CROP = 16

N_IMG_LAYERS = 13

N_CLASSES = 3

EARLY_STOP_PATIENCE = 10

N_MAX_EPOCHS = 500

DEFORESTATION_PATCHES_PREFIX = 'patches'
NO_DEFORESTATION_PATCHES_PREFIX = 'c_patches'
PREVIOUS_PREFIX = 'previous'
LABEL_PREFIX = 'label'

N_MAP_EVAL_PROCESSES = 7

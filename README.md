# A Transformer-based network for deforestation detection from bitemporal optical images

## Previous Deforestation Map Generator
Generate .tif previous deforestation temporal distance map. As older is the deforestation, the value is close to 0. As recent is the deforestation, the value is close to 1
```
usage: previous-def-gen.py [-h] [-b BASE_IMAGE] [-d DEFORESTATION_SHAPE] [-p PREVIOUS_DEFORESTATION_SHAPE] [-y YEAR] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_IMAGE, --base-image BASE_IMAGE
                        Path to optical tiff file as base to generate aligned labels
  -d DEFORESTATION_SHAPE, --deforestation-shape DEFORESTATION_SHAPE
                        Path to PRODES yearly deforestation shapefile (.shp)
  -p PREVIOUS_DEFORESTATION_SHAPE, --previous-deforestation-shape PREVIOUS_DEFORESTATION_SHAPE
                        Path to PRODES previous deforestation shapefile (.shp)
  -y YEAR, --year YEAR  Reference year to generate the temporal distance map. The higher value is 'year'-1.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to output label .tif folder
                        
 ```
## Label Generation
Generate .tif label file from PRODES deforestation shapefile.
```
usage: label-gen.py [-h] [-b BASE_IMAGE] [-d DEFORESTATION_SHAPE] [-p PREVIOUS_DEFORESTATION_SHAPE] [-w HYDROGRAPHY_SHAPE] [-n NO_FOREST_SHAPE] [-y YEAR]
                    [-o OUTPUT_PATH] [-i INNER_BUFFER] [-u OUTER_BUFFER]
                    
optional arguments:
  -h, --help            show this help message and exit
  -b BASE_IMAGE, --base-image BASE_IMAGE
                        Path to optical tiff file as base to generate aligned labels
  -d DEFORESTATION_SHAPE, --deforestation-shape DEFORESTATION_SHAPE
                        Path to PRODES yearly deforestation shapefile (.shp)
  -p PREVIOUS_DEFORESTATION_SHAPE, --previous-deforestation-shape PREVIOUS_DEFORESTATION_SHAPE
                        Path to PRODES previous deforestation shapefile (.shp)
  -w HYDROGRAPHY_SHAPE, --hydrography-shape HYDROGRAPHY_SHAPE
                        Path to PRODES hydrography shapefile (.shp)
  -n NO_FOREST_SHAPE, --no-forest-shape NO_FOREST_SHAPE
                        Path to PRODES no forest shapefile (.shp)
  -y YEAR, --year YEAR  Reference year to generate the labels
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to output label .tif folder
  -i INNER_BUFFER, --inner-buffer INNER_BUFFER
                        Inner buffer between deforestation and no deforestation to be ignored
  -u OUTER_BUFFER, --outer-buffer OUTER_BUFFER
                        Outer buffer between deforestation and no deforestation to be ignored
```
## Patches Preparation
prepare the files to be used in the training/testing steps
```
usage: prep-patches.py [-h] [--image-0 IMAGE_0] [--image-1 IMAGE_1] [--image-2 IMAGE_2] [-t TILES] [-f FILTER_OUTLIERS] [-m MIN_DEFORESTATION]

optional arguments:
  -h, --help            show this help message and exit
  --image-0 IMAGE_0     Path to optical image (.tif) file of year 0
  --image-1 IMAGE_1     Path to optical image (.tif) file of year 1
  --image-2 IMAGE_2     Path to optical image (.tif) file of year 2
  -t TILES, --tiles TILES
                        Path to the tiles file (.tif)
  -f FILTER_OUTLIERS, --filter-outliers FILTER_OUTLIERS
                        Filter image outliers option
  -m MIN_DEFORESTATION, --min-deforestation MIN_DEFORESTATION
                        Minimum deforestation
```
## Train models
Train NUMBER_MODELS models based in the same parameters
```
usage: train.py [-h] [-e EXPERIMENT] [-b BATCH_SIZE] [-n NUMBER_MODELS] [-x EXPERIMENTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        The number of the experiment
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The number of samples of each batch
  -n NUMBER_MODELS, --number-models NUMBER_MODELS
                        The number models to be trained from the scratch
  -x EXPERIMENTS_PATH, --experiments-path EXPERIMENTS_PATH
                        The patch to data generated by all experiments
```
## Prediction models
Predict NUMBER_MODELS models based in the same parameters
```
usage: predict.py [-h] [-e EXPERIMENT] [-b BATCH_SIZE] [-n NUMBER_MODELS] [-x EXPERIMENTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        The number of the experiment
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The number of samples of each batch
  -n NUMBER_MODELS, --number-models NUMBER_MODELS
                        The number models to be trained from the scratch
  -x EXPERIMENTS_PATH, --experiments-path EXPERIMENTS_PATH
                        The patch to data generated by all experiments
```
## F1-Score Evaluation
Evaluate F1-Score the models' prediction
```
usage: evaluate-f1.py [-h] [-e EXPERIMENT] [-n NUMBER_MODELS] [-x EXPERIMENTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        The number of the experiment
  -n NUMBER_MODELS, --number-models NUMBER_MODELS
                        The number models to be trained from the scratch
  -x EXPERIMENTS_PATH, --experiments-path EXPERIMENTS_PATH
                        The patch to data generated by all experiments
```
## Mean Average Prediction Evaluation
Evaluate mAP of the models' predictions
```
usage: evaluate-map.py [-h] [-e EXPERIMENT] [-n NUMBER_MODELS] [-x EXPERIMENTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        The number of the experiment
  -n NUMBER_MODELS, --number-models NUMBER_MODELS
                        The number models to be trained from the scratch
  -x EXPERIMENTS_PATH, --experiments-path EXPERIMENTS_PATH
                        The patch to data generated by all experiments
```

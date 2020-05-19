from pathlib import Path 
import pandas as pd 

# Path to main directory, will need to be set
ROOT = Path("/home/cooper/Documents/chicago_urban/mnp/cooper_prclz")

# Below this, paths will be automatically set based on ROOT
DATA_PATH = ROOT / "data"
GEOFABRIK_PATH = DATA_PATH / "input"
GEOJSON_PATH = DATA_PATH / "geojson"   

BLOCK_PATH = DATA_PATH / "blocks"
BLDGS_PATH = DATA_PATH / "buildings"
PARCELS_PATH = DATA_PATH / "parcels"
LINES_PATH = DATA_PATH / "lines"
COMPLEXITY_PATH = DATA_PATH / "complexity"

GADM_PATH = DATA_PATH / "GADM"
GADM_GEOJSON_PATH = DATA_PATH / "geojson_gadm"

TRANS_TABLE = pd.read_csv((ROOT / "data_processing" / 'country_codes.csv'))

all_paths = [BLOCK_PATH, GEOJSON_PATH, GADM_PATH, GADM_GEOJSON_PATH, 
            PARCELS_PATH, GEOFABRIK_PATH, COMPLEXITY_PATH, BLDGS_PATH]

# Create data dirs
for p in all_paths:
	p.mkdir(parents=True, exist_ok=True)
	

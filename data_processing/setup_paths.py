from pathlib import Path 

# Path to main directory, will need to be set
ROOT = Path("/home/cooper/Documents/chicago_urban/mnp/cooper_prclz")

# Below this, paths will be automatically set based on ROOT
DATA_PATH = ROOT / "data"
BLOCK_PATH = DATA_PATH / "blocks"
GEOJSON_PATH = DATA_PATH / "geojson"   
GADM_PATH = DATA_PATH / "GADM"
GADM_GEOJSON_PATH = DATA_PATH / "geojson_gadm"
GEOFABRIK_PATH = DATA_PATH / "input"

TRANS_TABLE = pd.read_csv(os.path.join(ROOT, "data_processing", 'country_codes.csv'))

# Create data dirs
for p in [BLOCK_PATH, GEOJSON_PATH, GADM_PATH, GADM_GEOJSON_PATH, GEOFABRIK_PATH]:
	p.mkdir(parents=True, exist_ok=True)
	

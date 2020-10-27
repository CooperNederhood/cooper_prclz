# Cleaned and commented code for Million Neighborhoods Project
## Data download and processing

#### 1. Download GADM data
The GADM data provides boundaries which we use to partition the globe into computationally feasible parts
From within data_processing/
```python
from prclz.data_processing import download_gadm
download_gadm.update_gadm_data(data_root = "/path/to/your/data/directory/")
```


#### 2. Download Geofabrik data
We use Geofabrik to get our OpenStreetMap raw data. Download this for all regions via the following command.
From within data_processing/
```python
from prclz.data_processing import fetch_geofabrik_data
fetch_geofabrik_data.update_geofabrik_data(data_root = "/path/to/your/data/directory/")
```

#### 3. Extract buildings and lines from the raw Geofabrik data [SATEJ]
The raw Geofabrik data is split into country-level files. This step creates a single "buildings" and a "lines" file for each country. The files are in "/data/geojson/"

#### 4. Block extraction [SATEJ]
Blocks are defined as regions fully circumscribed by roads or natural boundaries. Blocks are our most granular unit of analysis. This step extracts those blocks.

#### 5. Split the country-specific building files by GADM
Each country-level file is simply too huge for efficient computation. So, use the GADM boundaries to split the buildings files. This also functions as a data validation and QC point because along with the processed output in "/data/" the script will output country-level summaries about the matching of the OSM buildings with the GADM boundaries including list of non-matched buildings and a .png summary of the matching. 
From within data_processing/
```python
from prclz.data_processing import split_geojson
split_geojson.split_buildings(data_root = "/path/to/your/data/directory/")
```
NOTE: the default behavior is to process All the countries but if you want to process only one, then you can add the
3-letter country code contained in "country_codes.csv"
```python
from prclz.data_processing import split_geojson
split_geojson.split_buildings(data_root = "/path/to/your/data/directory/", gadm_name='DJI')
```

#### 6. Block complexity [SATEJ]

#### 7. Parcelization [NICO]


## Reblocking
There are options for reblocking depending on whether you want to reblock an entire country, just certain GADMs, and just certain blocks. 

##### Reblock entire GADM
```
python3 i_reblock.py --region Africa --gadm_name DJI --gadm DJI.1.1_1 --simplify
         
```
##### Process an entire GADM, but put specific blocks first, ahead of other blocks within the GADM
```
python3 i_reblock.py --region Africa --gadm_name DJI --gadm DJI.1.1_1 --blocks DJI.1.1_1_2 DJI.1.1_1_4 --simplify
```
##### Process only those specific blocks, not the entire GADM
```
python3 i_reblock.py --region Africa --gadm_name DJI --gadm DJI.1.1_1 --blocks DJI.1.1_1_2 DJI.1.1_1_4 --simplify --only_block_list
```

##### Because we have split out inputs by GADM, we can just reblock all the GADMs in a specific directory, essentially reblocking the entire country
```
python3 i_reblock.py --region Africa --gadm_name DJI  --gadm DJI.1.1_1 --simplify --from_dir ../data/buildings/Africa/DJI/
         
```


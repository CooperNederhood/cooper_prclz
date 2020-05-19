# Cleaned and commented code for Million Neighborhoods Project written by Cooper Nederhood
## Data download and processing

#### 1. Download GADM data
The GADM data provides boundaries which we use to partition the globe into computationally feasible parts
From within data_processing/
```
python3 download_gadm.py
```


#### 2. Download Geofabrik data
We use Geofabrik to get our OpenStreetMap raw data. Download this for all regions via the following command.
From within data_processing/
```
python3 fetch_geofabrik_data.py
```

#### 3. Extract buildings and lines from the raw Geofabrik data [SATEJ]
The raw Geofabrik data is split into country-level files. This step creates a single "buildings" and a "lines" file for each country. The files are in "/data/geojson/"

#### 4. Block extraction [SATEJ]
Blocks are defined as regions fully circumscribed by roads or natural boundaries. Blocks are our most granular unit of analysis. This step extracts those blocks.

#### 4. Split the country-specific building files by GADM
Each country-level file is simply too huge for efficient computation. So, use the GADM boundaries to split the buildings files. This also functions as a data validation and QC point because along with the processed output in "/data/" the script will output country-level summaries about the matching of the OSM buildings with the GADM boundaries including list of non-matched buildings and a .png summary of the matching. 
From within data_processing/
```
python3 split_geojson.py 
```
NOTE: the default behavior is to process All the countries but if you want to process only one, then you can add the
3-letter country code contained in "country_codes.csv"
```
python3 split_geojson.py --gadm_name DJI
```
## Reblocking
There are options for reblocking depending on whether you want to reblock an entire country, just certain GADMs, and just certain blocks. 

```
python3 i_reblock.py --region {Africa|Asia|Australia-Oceania|Central-America|Europe|North-America|South-America} \
                     --gadm_name DJI \
         
```

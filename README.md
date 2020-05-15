# Cleaned and commented code for Million Neighborhoods Project written by Cooper Nederhood

### 1. Download GADM data
The GADM data provides boundaries which we use to partition the globe into computationally feasible parts
From within data_processing/
```
python3 download_gadm.py
```


### 2. Download Geofabrik data
We use Geofabrik to get our OpenStreetMap raw data. Download this for all regions via the following command.
From within data_processing/
```
python3 fetch_geofabrik_data.py
```

### 3. Extract buildings and lines from the raw Geofabrik data
The raw Geofabrik data is split into country-level files. This step creates a "buildings" and a "lines" file for each country. The files are in "/data/geojson/"

### 4. Split the country-specific building files by GADM
Each country-level file is simply too huge for efficient computation. So, use the GADM boundaries to split the buildings files.
From within data_processing/
```
python3 split_geojson.py 
```
NOTE: the default behavior is to process All the countries but if you want to process only one, then you can add the
3-letter country code contained in "country_codes.csv"
```
python3 split_geojson.py --gadm_name DJI
```

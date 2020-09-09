import os 
import geopandas as gpd 
import pandas as pd 
from pathlib import Path 

from i_topology import PlanarGraph
from i_topology_utils import csv_to_geo
from i_reblock import add_buildings, clean_graph
from path_cost import FlexCost


# Should just import the setup_paths.py script (eventually)
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


def load_reblock_inputs(region: str, gadm_code: str, gadm: str):

    # Paths
    parcels_path = os.path.join(PARCELS_PATH, region, gadm_code, "parcels_{}.geojson".format(gadm))
    buildings_path = os.path.join(BLDGS_PATH, region, gadm_code, "buildings_{}.geojson".format(gadm))
    blocks_path = os.path.join(BLOCK_PATH, region, gadm_code, "blocks_{}.csv".format(gadm))

    # Load the files
    parcels_df = gpd.read_file(parcels_path)
    buildings_df = gpd.read_file(buildings_path)
    blocks_df = csv_to_geo(blocks_path)
    blocks_df.rename(columns={'block_geom': 'geometry'}, inplace=True)
    blocks_df = blocks_df[['geometry', 'block_id']]
    blocks_df = gpd.GeoDataFrame(blocks_df, geometry='geometry')

    # Map buildings to a block
    # Convert buildings to centroids
    buildings_df['buildings'] = buildings_df['geometry'].centroid
    buildings_df.set_geometry('buildings', inplace=True)

    # We want to map each building to a given block to then map the buildings to a parcel
    buildings_df = gpd.sjoin(buildings_df[['buildings', 'osm_id', 'geometry']], blocks_df, how='left', op='within')
    buildings_df = buildings_df[['buildings', 'block_id', 'geometry']].groupby('block_id').agg(list)
    buildings_df['building_count'] = buildings_df['buildings'].apply(lambda x: len(x))
    buildings_df.reset_index(inplace=True)

    return parcels_df, buildings_df, blocks_df 


def reblock_gadm(region, gadm_code, gadm, simplify, cost_fn, block_list=None):

    parcels, buildings, blocks = load_reblock_inputs(region, gadm_code, gadm)

    if block_list is None:
        block_list = blocks['block_id'].values

    for block_id in block_list:

        # (0) Get data for the block
        parcel_geom = parcels[parcels['block_id']==block_id]['geometry'].iloc[0]
        building_list = buildings[buildings['block_id']==block_id]['buildings'].iloc[0]
        block_geom = blocks[blocks['block_id']==block_id]['geometry'].iloc[0]

        # (1) Convert parcel geometry to planar graph
        planar_graph = PlanarGraph.multilinestring_to_planar_graph(parcel_geom)

        # (2) Add building centroids to the planar graph
        bldg_tuples = [list(b.coords)[0] for b in building_list]
        planar_graph = add_buildings(planar_graph, bldg_tuples)

        # (3) Clean the graph if its disconnected
        planar_graph, num_components = clean_graph(planar_graph)

        # (4) Update the planar graph if the cost_fn needs it
        if cost_fn.lambda_turn_angle > 0:
            planar_graph.set_node_angles()
        if cost_fn.lambda_width > 0:
            bldg_polys = buildings[buildings['block_id']==block_id]['geometry'].iloc[0]
            planar_graph.set_edge_width(bldg_polys)

        # (5) Do steiner approximation
        planar_graph.flex_steiner_tree_approx(cost_fn = cost_fn)


# #cost_fn = reblock2.FlexCost(lambda_width=1.0,lambda_degree=200., lambda_turn_angle=2.)
# cost_fn = reblock2.FlexCost()
# parcels, buildings, blocks = reblock2.load_reblock_inputs(region, gadm_code, gadm)
# planar_graph = PlanarGraph.multilinestring_to_planar_graph(parcel_geom)

# bldg_tuples = [list(b.coords)[0] for b in building_list]
# planar_graph = reblock2.add_buildings(planar_graph, bldg_tuples)

# planar_graph.set_node_angles()
# bldg_polys = buildings[buildings['block_id']==block_id]['geometry'].iloc[0]
# planar_graph.set_edge_width(bldg_polys)

# planar_graph, num_components = reblock2.clean_graph(planar_graph)


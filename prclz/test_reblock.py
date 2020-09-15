import i_topology
import geopandas as gpd 
import reblock2
from path_cost import FlexCost
import copy 
from shapely.geometry import LineString, MultiPolygon, Polygon, MultiLineString, Point, MultiPoint, LineString

def make_square(lower_left_pt, w=1):
    pts = []
    x,y = lower_left_pt
    i,j = 0,0   
    pts.append((x + w*i, y + w*j))

    i,j = 1,0   
    pts.append((x + w*i, y + w*j))

    i,j = 1,1   
    pts.append((x + w*i, y + w*j))

    i,j = 0,1   
    pts.append((x + w*i, y + w*j))
    pts.append(lower_left_pt)
    return LineString(pts)

def make_grid(lower_left_pt, h=3, w=3, delta=1):
    multi = []
    for i in range(w):
        for j in range(h):
            multi.append(make_square((i,j), delta))
    return multi 

def create_test_grid(n):
    multi = []
    for i in range(n):
        for j in range(n):
            multi.append(make_square((i,j), 1))
    return i_topology.PlanarGraph.multilinestring_to_planar_graph(multi)

def create_connected_grid():
    linestrings = []
    linestrings.extend(make_grid((0,0), h=3,w=3,delta=1))
    linestrings.extend(make_grid((6,0), h=3,w=3,delta=1))
    connector = LineString([(3,0), (4,0), (5,0), (6,0)])
    linestrings.append(connector)
    g = i_topology.PlanarGraph.multilinestring_to_planar_graph(linestrings)
    g.es['width'] = 5
    i = 1
    for e in g.es[[24, 25, 26]]:
        e['width'] = i
        i += 1
    return g 

def get_test_grids():
    grid = create_test_grid(2) 
    points = [
        (2, 2),
        (0, 0.2),
        (1.8, 1),
        (0.8, 2),
    ]

    grid2 = create_test_grid(2) 
    grid3 = create_test_grid(2) 
    grid4 = grid 
    for i, pt in enumerate(points):
        if i < 2:
            grid2.add_node_to_closest_edge(pt, terminal=True)
            grid3.add_node_to_closest_edge(pt, terminal=True)
            grid4.add_node_to_closest_edge(pt, terminal=True)
        elif i < 3:
            grid3.add_node_to_closest_edge(pt, terminal=True)
            grid4.add_node_to_closest_edge(pt, terminal=True)
        elif i < 4:
            grid4.add_node_to_closest_edge(pt, terminal=True)
        #grid.add_node_to_closest_edge(pt, terminal=True)
    grids = {2: grid2, 3: grid3, 4: grid4}
    return grids 

# # (1) Original reblocking
# test_grids = get_test_grids()
# for n, grid in test_grids.items():
#     grid.steiner_tree_approx()
#     p = "./test_dir/orig_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (2) Flex reblocking but with default settings (should match #1)
# test_grids = get_test_grids()
# default_cost_fn = FlexCost()
# for n, grid in test_grids.items():
#     grid.flex_steiner_tree_approx(cost_fn = default_cost_fn)
#     p = "./test_dir/flex_baseline_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (3) Flex reblocking -- penalize turns
# test_grids = get_test_grids()
# noturn_cost_fn = FlexCost(lambda_turn_angle=2.)
# for n, grid in test_grids.items():
#     grid.set_node_angles()
#     grid.flex_steiner_tree_approx(cost_fn = noturn_cost_fn)
#     p = "./test_dir/flex_no-turns_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (4) Flex reblocking -- penalize non-4 way compatible intersections
# test_grids = get_test_grids()
# fourway_cost_fn = FlexCost(lambda_degree=200., lambda_turn_angle=2.)
# for n, grid in test_grids.items():
#     grid.set_node_angles()
#     grid.flex_steiner_tree_approx(cost_fn = fourway_cost_fn)
#     p = "./test_dir/flex_fourways_n{}.png".format(n)
#     grid.plot_reblock(p)

# # (5) Flex reblocking -- add 'houses' and factor in width
# test_grids = get_test_grids()
# ls = make_square(lower_left_pt=(0.5, 1.1), w=0.2)
# bldg = [Polygon(ls)]
# width_cost_fn = FlexCost(lambda_width=1.0)
# for n, grid in test_grids.items():
#     grid.set_edge_width(bldg)
#     grid.flex_steiner_tree_approx(cost_fn = width_cost_fn)
#     p = "./test_dir/flex_width_n{}.png".format(n)
#     grid.plot_reblock(p)



# # (6) Test on DJI data
def plot_dji_test(block, output_filename):
    block_parcel = parcels[parcels['block_id']==block]
    block_bldgs = bldgs
    
    ax = block_parcel.plot(color='black', alpha=0.3)
    ax = block_bldgs.plot(color='black', ax=ax)
    reblock_new = reblock_data[reblock_data['line_type']=='new']
    reblock_existing = reblock_data[reblock_data['line_type']=='existing']

    ax = reblock_new.plot(color='green', ax=ax)
    ax = reblock_existing.plot(color='red', ax=ax)
    return ax

region = 'Africa'
gadm_code = 'DJI'
gadm = 'DJI.1.1_1'
block_list = ['DJI.1.1_1_1']

cost_fn = FlexCost()
reblock_data, parcels, buildings, blocks =  reblock2.reblock_gadm(region, gadm_code, gadm, cost_fn, block_list=block_list)
bldgs = gpd.read_file('../data/buildings/Africa/DJI/buildings_DJI.1.1_1.geojson')

ax = plot_dji_test('DJI.1.1_1_1', 'dji_test.png')

# parcels, buildings, blocks = reblock2.load_reblock_inputs(region, gadm_code, gadm)

# block_id = buildings['block_id'].iloc[0]
# parcel_geom = parcels[parcels['block_id']==block_id]['geometry'].iloc[0]
# building_list = buildings[buildings['block_id']==block_id]['buildings'].iloc[0]
# block_geom = blocks[blocks['block_id']==block_id]['geometry'].iloc[0]

# planar_graph = i_topology.PlanarGraph.multilinestring_to_planar_graph(parcel_geom)
# bldg_tuples = [list(b.coords)[0] for b in building_list]
# planar_graph = reblock2.add_buildings(planar_graph, bldg_tuples)
# planar_graph.set_node_angles()
# bldg_polys = buildings[buildings['block_id']==block_id]['geometry'].iloc[0]
# planar_graph.set_edge_width(bldg_polys)
# planar_graph, num_components = reblock2.clean_graph(planar_graph)
# planar_graph.flex_steiner_tree_approx(cost_fn = cost_fn)

# new_steiner, existing_steiner = planar_graph.get_steiner_linestrings(expand=False)


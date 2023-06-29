import folium
import osmnx as ox
import alphashape
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
import requests
import utm
import numpy as np
import geojson
from tqdm import tqdm
import pandas as pd


def get_utm(row, utm_info: dict):
    res = utm.from_latlon(
        float(row[utm_info["lat_row"]]),
        float(row[utm_info["long_row"]]),
        utm_info["zone_paramter_1"],
        utm_info["zone_parameter_2"],
    )

    return res[0], res[1]


def get_utm_array(lat, longt, utm_info: dict):
    res = utm.from_latlon(
        lat, longt, utm_info["zone_paramter_1"], utm_info["zone_parameter_2"]
    )

    return res


def filter_grid(grid, address):
    G = ox.graph_from_place(address)
    nodes, edges = ox.graph_to_gdfs(G)

    test_nodes = [[x, y] for x, y in zip(nodes["geometry"].y, nodes["geometry"].x)]

    import alphashape

    alpha_shape = alphashape.alphashape(test_nodes, alpha=0)
    area_of_interest = gpd.GeoDataFrame(
        index=[0], crs="EPSG:20791", geometry=[alpha_shape]
    )

    grid["geos"] = [
        Point(x, y) for x, y in zip(grid["center_lat"], grid["center_long"])
    ]
    comscore_grid_gdf = gpd.GeoDataFrame(crs="EPSG:20791", geometry=grid["geos"])

    points_in_area_of_interest = gpd.sjoin(area_of_interest, comscore_grid_gdf)

    df = pd.merge(
        grid, points_in_area_of_interest, left_index=True, right_on="index_right"
    )

    return df


def accessibility_score(points_df, non_intersection_col_list, intersection_col_list):
    obstacle_cols = non_intersection_col_list
    intersection_obstacle_cols = intersection_col_list

    for col in intersection_obstacle_cols:
        points_df[f"{col}_quantile"] = pd.qcut(
            points_df[col], 100, labels=np.arange(0, 100)
        ).astype(int)

    points_df["elevation_rank"] = points_df["elevation_rank"].astype(int)

    col_list = [f"{x}_quantile" for x in obstacle_cols] + ["elevation_rank"]
    intersection_col_list = [
        f"{x}_quantile" for x in intersection_obstacle_cols
    ]  # + ['elevation_rank']

    inaccessibility_scores = []

    for idx, point in points_df.iterrows():
        if point["is_intersection"] == False:
            inaccessibility_scores.append(max(point[col_list]))

        else:
            inaccessibility_scores.append(max(point[intersection_col_list]))

    points_df["inaccessibility_score"] = inaccessibility_scores

    points_df["accessibility_score"] = (100 - points_df["inaccessibility_score"]) / 100

    return points_df


def get_drive_map_intersections(address, utm_info):
    G = ox.graph_from_place(address, network_type="drive")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes.crs = "epsg:4326"
    gdf_edges.crs = "epsg:4326"
    gdf_nodes = gdf_nodes.to_crs("EPSG:27429")
    gdf_edges = gdf_edges.to_crs("EPSG:27429")

    # make nodes consistent...
    gdf_nodes["x"] = gdf_nodes["geometry"].x
    gdf_nodes["y"] = gdf_nodes["geometry"].y

    G = ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)

    intersections = ox.consolidate_intersections(
        G, tolerance=15, dead_ends=False, rebuild_graph=False
    )

    intersections.crs = "EPSG:27429"
    intersections = intersections.to_crs(4326)

    intersections = pd.DataFrame(intersections, columns=["geos"])
    intersections[["lat", "long"]] = [[geo.y, geo.x] for geo in intersections["geos"]]
    intersections[["x_utm", "y_utm"]] = intersections.apply(
        lambda x: get_utm(x, utm_info), axis=1, result_type="expand"
    )

    return intersections


def define_intersections(address, points_df, utm_info):
    intersections = get_drive_map_intersections(address, utm_info)

    points_df["geos_utm"] = [
        Point(row["x_utm"], row["y_utm"]) for id, row in points_df.iterrows()
    ]
    intersections["geos_utm"] = [
        Point(row["x_utm"], row["y_utm"]) for id, row in intersections.iterrows()
    ]

    intersection_gdf = gpd.GeoDataFrame(geometry=intersections["geos_utm"])
    data_lean_gdf = gpd.GeoDataFrame(geometry=points_df["geos_utm"])

    found_inters = gpd.sjoin_nearest(
        intersection_gdf, data_lean_gdf, distance_col="distances"
    )
    found_inters = found_inters[found_inters["distances"] < 20]

    points_df["is_intersection"] = False
    for point in found_inters["index_right"]:
        points_df["is_intersection"].iloc[point] = True

    return points_df


def get_drive_map_midpoints_elevations(address, utm_info, url):
    cf = '["highway"~"path|footway|steps|cycleway|primary"]'

    G = ox.graph_from_place(address, network_type="drive", custom_filter=cf)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes.crs = "epsg:4326"
    gdf_edges.crs = "epsg:4326"
    gdf_nodes = gdf_nodes.to_crs("EPSG:27429")
    gdf_edges = gdf_edges.to_crs("EPSG:27429")

    # make nodes consistent...
    gdf_nodes["x"] = gdf_nodes["geometry"].x
    gdf_nodes["y"] = gdf_nodes["geometry"].y

    streets = gdf_edges["geometry"]
    streets.crs = "EPSG:27429"
    streets = streets.to_crs(4326)

    midpoints = []

    for segment in tqdm(streets):
        segment_points = ox.utils_geo.interpolate_points(segment, segment.length / 4)

        for point in segment_points:
            midpoints.append(Point(point[0], point[1]))

    midpoint_geos = gpd.GeoSeries(midpoints)
    midpoints = (
        gpd.GeoDataFrame(data=midpoint_geos)
        .drop_duplicates()
        .reset_index()
        .rename(columns={0: "geo"})
        .drop(columns="index")
    )

    coord_array = [[geo.y, geo.x] for geo in midpoints["geo"]]
    midpoints["lat"] = [x[0] for x in coord_array]
    midpoints["long"] = [x[1] for x in coord_array]
    midpoints["elevations"] = [
        get_point_elevation(geo.y, geo.x, url) for geo in tqdm(midpoints["geo"])
    ]
    midpoints[["x_utm", "y_utm"]] = midpoints.apply(
        lambda x: get_utm(x, utm_info), axis=1, result_type="expand"
    )

    return midpoints


def get_variance_rank(address, points_df, utm_info, url):
    midpoints = get_drive_map_midpoints_elevations(
        address=address, utm_info=utm_info, url=url
    )

    points = midpoints[["x_utm", "y_utm"]].values
    tree = cKDTree(points)

    points_df[["x_utm", "y_utm"]] = points_df.apply(
        lambda x: get_utm(x, utm_info), axis=1, result_type="expand"
    )

    # Indexes the grid in a KD Tree
    tree_grid = cKDTree(points_df[["x_utm", "y_utm"]].values)
    tree_grid_centers = points_df[["x_utm", "y_utm"]].values

    neighbours = tree.query_ball_point(tree_grid_centers, r=300)

    points_df["neighbours"] = neighbours
    points_df["nbr_neighbours"] = [len(x) for x in neighbours]

    within_radius = tree_grid.sparse_distance_matrix(tree, max_distance=300).todense()
    transposed_idxs = np.transpose(within_radius.nonzero())
    distances = within_radius[within_radius.nonzero()]

    # Get nearest neighbours
    nns_lisbon_ids = points_df.iloc[transposed_idxs[:, 0]][["x_utm", "y_utm"]]
    nns = midpoints.iloc[
        transposed_idxs[:, 1]
    ]  # [aggregator_cols] # [['ad_id', 'ad_unitprice', 'ad_typology', 'ad_operation', 'date_ranges']]

    ret = pd.concat(
        [nns_lisbon_ids.reset_index(drop=True), nns.reset_index(drop=True)], axis=1
    )

    ret.columns = [
        "x_utm_grid",
        "y_utm_grid",
        "geo",
        "lat",
        "long",
        "elevations",
        "x_utm",
        "y_utm",
    ]
    std_elevation_grouped = ret.groupby(["x_utm_grid", "y_utm_grid"])[
        "elevations"
    ].std()
    data_per_loc_merge = points_df.merge(
        std_elevation_grouped,
        left_on=["x_utm", "y_utm"],
        right_on=["x_utm_grid", "y_utm_grid"],
    )
    data_per_loc_merge["elevation_rank"] = pd.qcut(
        data_per_loc_merge["elevations"],
        100,
        labels=np.arange(0, 99),
        duplicates="drop",
    )
    data_per_loc_merge = data_per_loc_merge[data_per_loc_merge["nbr_neighbours"] > 1]
    points_df = data_per_loc_merge.reset_index()

    return points_df


def get_point_elevation(lat, longt, url):
    coord = f"{lat},{longt}"

    req = requests.get(f"{url}{coord}")

    return req.json()["results"][0]["elevation"]


def get_dist_to_cycle_lane(geojson_path, points_df, utm_info):
    with open(geojson_path) as f:
        gj = geojson.load(f)

    cycleways = [cycleway["geometry"]["coordinates"] for cycleway in gj["features"]]
    cycle_points = []
    for cycleway in cycleways:
        for point in cycleway:
            cycle_points.append(point)

    cycle_points_geos = []
    for x in cycle_points:
        try:
            cycle_points_geos.append(Point(x[0], x[1]))
        except:
            pass

    points_df["geo"] = [
        Point(row["x_utm"], row["y_utm"]) for idx, row in points_df.iterrows()
    ]
    data_lean_gdf = gpd.GeoDataFrame(points_df, geometry="geo")
    cycle_point_gdf = gpd.GeoDataFrame(geometry=cycle_points_geos)
    coord_array = [[geo.y, geo.x] for geo in cycle_point_gdf["geometry"]]
    cycle_point_gdf["lat"] = [x[0] for x in coord_array]
    cycle_point_gdf["long"] = [x[1] for x in coord_array]
    cycle_point_gdf[["x_utm", "y_utm"]] = cycle_point_gdf.apply(
        lambda x: get_utm(x, utm_info), axis=1, result_type="expand"
    )
    cycle_point_gdf["geometry"] = [
        Point(point["x_utm"], point["y_utm"])
        for idx, point in cycle_point_gdf.iterrows()
    ]
    near = data_lean_gdf.sjoin_nearest(cycle_point_gdf, distance_col="dist_to_cycle")

    points_df["dist_to_cycle"] = 0
    for idx, point in near.iterrows():
        points_df["dist_to_cycle"].iloc[idx] = point["dist_to_cycle"]

    return points_df, cycle_point_gdf


def grid_maker(
    boundaries: dict, overlap_factor: float = 1, ratio: float = 500 / 110500
):
    # there is no sobreposition, assuming that 1 degree (lat/long) is roughly
    # 110.5km and that 600 pixels will correspond to 600m (ignoring the conversion
    # of 1.19 factor)

    # this is with no sobreposition
    vertical_tiles = round((boundaries["north"] - boundaries["south"]) / ratio)
    horizontal_tiles = abs(round((boundaries["west"] - boundaries["east"]) / ratio))
    print("vertical_tiles:", vertical_tiles, "horizontal_tiles:", horizontal_tiles)

    # ading a factor to guarantee a overlap
    vertical_tiles = round(vertical_tiles * overlap_factor)
    horizontal_tiles = round(horizontal_tiles * overlap_factor)
    print("vertical_tiles:", vertical_tiles, "horizontal_tiles:", horizontal_tiles)

    lats_unique = np.linspace(boundaries["north"], boundaries["south"], vertical_tiles)
    longs_unique = np.linspace(boundaries["west"], boundaries["east"], horizontal_tiles)

    longs, lats = np.meshgrid(longs_unique, lats_unique)

    return longs, lats


def create_community_score_from_pois(grid, df_pois, radius_in_meters):
    # Indexes the prediction points in a KDTree
    points = df_pois[["x_utm", "y_utm"]].values
    tree = cKDTree(points)

    # Indexes the grid in a KD Tree
    tree_grid = cKDTree(grid[["x_utm", "y_utm"]].values)
    tree_grid_centers = grid[["x_utm", "y_utm"]].values

    neighbours = tree.query_ball_point(tree_grid_centers, r=radius_in_meters)

    # Calculates a sparse matrix of the points within a 5 km distance which is weighted down
    # as the distance increases - useful for getting a score
    # such as the density of stores, ...
    within_radius = tree_grid.sparse_distance_matrix(
        tree, max_distance=radius_in_meters
    )
    within_radius = within_radius.todense()
    dists = np.sum(np.divide(1, within_radius, where=within_radius != 0), axis=1)

    # Ranks them by the distance
    grid["tmp"] = dists
    grid["tmp_rank"] = grid["tmp"].rank(pct=True)
    comscore_grid = grid.drop(columns="tmp")

    for i in range(len(neighbours)):
        dist_matrix = np.asarray(within_radius[i]).ravel()[neighbours[i]]

    comscore_grid["neighbourhood_accessibility"] = 0
    comscore_grid["neighbour_size"] = 0
    for idx, neighbour in tqdm(enumerate(neighbours)):
        if neighbour:
            dist_matrix = 1 / np.asarray(within_radius[idx]).ravel()[neighbours[idx]]

            neighbourhood_scores = np.sum(
                [df_pois["accessibility_score"].iloc[x] for x in neighbour]
                * dist_matrix
            ) / np.sum(dist_matrix)

            comscore_grid["neighbourhood_accessibility"].iloc[
                idx
            ] = neighbourhood_scores
            comscore_grid["neighbour_size"].iloc[idx] = len(neighbour)

    return comscore_grid, neighbours, within_radius


def get_interactive_map(start_location, coords):
    m = folium.Map(location=start_location, zoom_start=13)

    for coord in coords:
        m.add_child(
            folium.CircleMarker(
                location=coord,
                radius=2,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
            ).add_to(m)
        )

    return m


def address_scores_from_grid(addresses, grid):
    zone_scores = []

    for address in addresses:
        # map
        G = ox.graph_from_place(address)

        nodes, edges = ox.graph_to_gdfs(G)

        test_nodes = [[x, y] for x, y in zip(nodes["geometry"].y, nodes["geometry"].x)]
        alpha_shape = alphashape.alphashape(test_nodes, alpha=0)
        area_of_interest = gpd.GeoDataFrame(
            index=[0], crs="EPSG:20791", geometry=[alpha_shape]
        )

        grid["geos"] = [
            Point(x, y) for x, y in zip(grid["center_lat"], grid["center_long"])
        ]

        comscore_grid_gdf = gpd.GeoDataFrame(crs="EPSG:20791", geometry=grid["geos"])

        points_in_area_of_interest = gpd.sjoin(area_of_interest, comscore_grid_gdf)

        scores = [
            grid["neighbourhood_accessibility"].iloc[x]
            for x in points_in_area_of_interest["index_right"]
        ]

        for score in scores:
            zone_scores.append(score)

    return zone_scores

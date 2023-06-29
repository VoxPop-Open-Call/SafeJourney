from random import sample
import osmnx as ox
import os
import geopandas
from shapely.geometry import Point
from osmnx import graph_to_gdfs, graph_from_address
from shapely.geometry import Point
import fnmatch
import tqdm
import argparse
from utils import get_multi_street

from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()

PARSER = argparse.ArgumentParser(description="Script to get street and sattelite images from geo location node")

PARSER.add_argument(
    "--address",
    metavar="ad",
    type=str,
    nargs="?",
    help="Address of the location to get geometries from",
)

PARSER.add_argument(
    "--geo-type",
    metavar="gt",
    type=str,
    nargs="?",
    help="Type of geometry to retrieve (crosswalk, intersection or midpoint)",
)

PARSER.add_argument(
    "--base-path",
    metavar="bp",
    type=str,
    nargs="?",
    help="Folder to store images",
)

PARSER.add_argument(
    "--number-of-nodes",
    metavar="nn",
    type=int,
    nargs="?",
    help="Number of nodes to query",
)

ARGS = PARSER.parse_args()

address = ARGS.address
geo_type = ARGS.geo_type
sample_size = ARGS.number_of_nodes
base_path = ARGS.base_path

GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/staticmap"
GOOGLE_STREET_VIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"
API_KEY = os.getenv("GOOGLE_STREET_VIEW_KEY")

print("=== STARTING ===")
print(f"Getting {sample_size} nodes of type {geo_type} from location: {address}.")
print(f"Expected return: {sample_size*5} images.")
print(f"Image path = {base_path}")

if geo_type == "crosswalk":

  # List key-value pairs for tags
  tags = {'highway': 'crossing'}
  street_crossing = ox.geometries_from_place(address,tags)
  sample_geos = street_crossing['geometry']

elif geo_type == "intersection":

  G = ox.graph_from_place(address, network_type='drive')
  gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
  gdf_nodes.crs = "epsg:4326"
  gdf_edges.crs = "epsg:4326"
  gdf_nodes = gdf_nodes.to_crs("EPSG:27429")
  gdf_edges = gdf_edges.to_crs("EPSG:27429")

  # make nodes consistent...
  gdf_nodes["x"] = gdf_nodes["geometry"].x
  gdf_nodes["y"] = gdf_nodes["geometry"].y

  G = ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)

  intersections = ox.consolidate_intersections(G, tolerance=15, dead_ends=False, rebuild_graph=False)

  intersections.crs = "EPSG:27429"
  intersections = intersections.to_crs(4326)
  sample_geos = intersections

elif geo_type == "midpoint":

  G = ox.graph_from_place(address, network_type='drive')

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

  for segment in streets:
      segment_points = ox.utils_geo.interpolate_points(segment, segment.length/2)

      for point in segment_points:
          midpoints.append(Point(point[0], point[1]))

  midpoint_geos = geopandas.GeoSeries(midpoints)
  sample_geos = midpoint_geos



# Setup =============================================


degs = [0,90,180,270]
headings = {0, 90, 180, 270}
fov = 120
pitch = -40

sample_geos = sample_geos.sample(n=sample_size, random_state=42)
print(len(sample_geos))


# Fetch data ====================================
for idx, geo in enumerate(sample_geos):
 
  location_path = base_path+f"/{geo.x}_{geo.y}" 

  try:
    os.mkdir(location_path)
  except:
    print("Folder already found")
    # if folder.count_pngs >=5: skip iteration
    count = len(fnmatch.filter(os.listdir(location_path), '*.*'))

    #print('File Count:': count)
    if count >= 5:
      continue

  test_images = get_multi_street(geo.y, geo.x, headings, pitch, fov, API_KEY, GOOGLE_STREET_VIEW_API_URL, GOOGLE_MAPS_API_URL)

  for jdx, image in enumerate(test_images):
    if jdx < 4:
      image = image.save(location_path+f"/{geo.x}_{geo.y}_{degs[jdx]}.png")
    #else:
      #image = image.save(location_path+f"/{geo.x}_{geo.y}_satellite.png")

print("=== FINISHED ===")
# VoxPop - SafeJourney by NILG.AI 
<img src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" data-canonical-src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" width="900" height="250" />


The existence of multiple barriers to mobility deprive a lot of wheelchair users of following a certain path on the street. Currently, wheelchair users need to carefully plan the path they need to take ahead of time. 

However, information about accessibility of public spaces is not easily accessible nor updated, making wheelchair users take risky decisions such as shifting to the road to surpass a certain obstacle.

SafeJourney uses satellite, street and geographical data to calculate an accessibility index for users of electrical and mechanical wheelchairs, by calculating:

- If the sidewalk is irregular or not
- If the crosswalk has sufficient width
- If the crosswalk has a ramp
- The slope of the region

… among other indicators

This accessibility index is going to be calculated for thousands of points in the city of Lisbon, providing information about what regions are friendlier for users in wheelchairs. On the long term, this solution can be used not only by wheelchair users directly but by service providers (e.g. restaurants/real estate) to give more detailed information about the region to these users.

## Project deliverables
The resulting data and trained models used in this project can be found in our google drive. The following items are available:
 
 - [Report creation with accessibility index](https://drive.google.com/file/d/1CRvu5JxV80WyZJukyj8KrbL5rUPDz1B0/view?usp=sharing)
 - [Heatmap of city with accessibility score](https://drive.google.com/file/d/1yveIcSuKKenoy8ZCRMQYAr_5n8qV2lII/view?usp=sharing)
 - [Demo application landing](https://nilg.ai/safejourney)
 - [Annotated data for the crosswalks and sidewalks](https://drive.google.com/file/d/1T8SsasRYjv03d9j0nOlv2GnXImh20VWj/view?usp=sharing)
 - [Annotated data for the temporary obstacles](https://drive.google.com/file/d/1k_7LGPD19b274uG7TTOW8v033F8OQPuD/view?usp=sharing)
 - [Grid scores (source of the accessibility score)](https://drive.google.com/file/d/14Hz-LR7WJgfHs42fLqXzWXnngZk6zuD9/view?usp=sharing)
 - [Individual scores for the extracted points](https://drive.google.com/file/d/1rZZ5SpJFaS3SDnGfkSsVzE0DbVf5cJ0W/view?usp=sharing)

## Model Preparation
Here we will describe how we performed each task to prepare and train our computer vision model, and how you can replicate it.
### Obtaining relevant geospatial data

We use OpenStreetMaps (OSM) to obtain data for the region we want. Using their tag system we can obtain different types of geospatial information. In this project, we focused on obtaining crosswalks, intersections, and points along the drive map of Lisbon. Here's an example of extracting latitudes and longitudes of crosswalks in Lisbon.

```python
# an example of obtaining crosswalk data for a given location
import omnx as ox
address = 'Lisbon, Portugal'
# List key-value pairs for tags
tags = {'highway': 'crossing'}
street_crossing = ox.geometries_from_place(address,tags)
sample_geos = street_crossing['geometry']

```

### Data Acquisition

We train our model with images we get from Google Maps Street View. To perform data scraping you can use the script (point to script). You of course need a google street view API key of your own. Here is an example of a bash command to run the script:

```bash
python fetch_images.py --address 'Lisbon, Portugal' --geo-type 'crosswalk' --base-path 'data/lisbon_crosswalks' --number-of-nodes 2000 --google-street-view-key 'your api key' 
```

For each node, the script will retrieve 4 geopoint images from 4 different angles (0, 90, 180, 270). We also use a FOV of 120 and a pitch of -40. These can be changed in the script.

### Deploying a labeling infrastructure for the images

For labeling, we used LabelStudio and the Labelstudio python SDK to create several projects for image labeling. To set up your own labeling infrastructure you must first create an EC2 container where you can deploy your docker image. Then you have to clone the [labelstudio repo](https://github.com/heartexlabs/label-studio) to the machine. Then build your docker image and compose it. To test your connection, run:

```python
# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'insert server url' 
API_KEY = "insert API key"

# Connect to the Label Studio API and check the connection
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

# Check all the projects in Label Studio
projects = ls.list_projects()﻿
# Access the project you want
project = ls.get_project("your project ID here") #for example ls.get_project(1)

# Encode the file you want to upload
def read_and_encode(filename):
    with open(filename, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return encoded_string.decode("utf-8")


encoded_file = read_and_encode("file_path")

# Create a task and upload it 
task = {
            "data": {
                "image": f"data:image/jpeg;base64,{encoded_file}",
                "...": ...
                }
            }
            
project.import_tasks(
    [task]
)
```

Depending on the container type and capacity, sequentially uploading thousands of tasks to labelstudio might break the service. To avoid this,  our [script](upload_to_labelstudio.py) uploads 100 tasks at a time with a one minute sleep in between. 

### Model Training

With annotated data you can now train your model. We trained a multitask image classification model with a MobileNetV2 architecture, pre-trained on ImageNet. The tasks were to identify cars/scooters on top of sidewalks, sidewalk width, and crosswalk level.
Then we ran inference for every image we had retrieved from the google maps street view (roughly 60 000 images).

You can find all of the code you need to train and test the model in the [model utils](model_utils.py) script. You can also run our [model training script](train_classifier.py). You will need to provide it a yaml config file like [this one](config.yaml).

```python
python train_classifier.py --config-path 'path to config file'
```

## Extractions of points of interest

Once you have your predictions, you can start calculating metrics for it, such as proximity to cycle lanes, elevations, and whether or not they are an intersection. All of the functions used in these examples are available in the [geo utils](geo_utils.py) script.

To distance calculation purposes, we use the UTM system. You will need to provide some functions with a python dict containing the UTM info of the region you are analyzing, so we can convert the coordinates to the UTM system.

```python
import utm

utm_info = {
    'lat_row': 'lat',
    'long_row': 'long',
    'zone_paramter_1': 29,
    'zone_parameter_2': 'N'
}
```

### 1 - Cycle lanes
To extract cycle lane data used for this project, you need to go to the [Lisboa Aberta website](https://dados.cm-lisboa.pt/dataset/rede-ciclavel) and download their data. Then, to associate a distance to cycle lane to each point you can run:

```python
geojson_path = "your path"
points_df = #your annotated data

# get a df with the distances associated to each point, and also the cycle lane geodataframe
points_df, cycle_point_gdf = get_dist_to_cycle_lane(geojson_path, points_df) 
```

### 2 - Elevation

We obtain the elevation for each point using the Open Elevation API. If the API is not working, you can always deploy the Open Elevation tool onto a Docker Container, and 
host it locally.

```python
point_elevation = get_point_elevation(your_lat, your_long, your_url)
```

To get an intuition about how hilly a segment of road is, we calculate the variance of elevations around each point, by collecting the elevations of other road points in a given radius. For that, we need to obtain as many relevant points as we can get. We use OSM to give us the road map of Lisbon, and from that, we extract each point present in the line segments and calculate their elevation. We then rank the variances of each point from 0-100 using pandas qcut, which will give us our elevation score. This score takes into account the distance between the given point and the surrounding points, giving less impact to the ones further away. We compacted all of these operations into a function you can readily use.

```python
points_df = get_variance_rank(address, points_df, utm_info, url):
```

### 3 - Crosswalks

Later we will be basing our scores on how good a crosswalk is, however this score will only be relevant in places where there should be a crosswalk (e.g. intersections). This means the score will depend on whether or not a location is an intersection. To get this information, we obtain all the intersections from OSM, and then join them with our annotated data. The result is a column of boolean values, which can be obtained by calling the function:

```python
points_df = define_intersections(address, points_df, utm_info):
```

## Score Calculation

Once we have our model predictions and the rest of the information previously described, we can aggregate all of those values in a single score - our accessibility score. To get the score you can simply call:

```python
points_df = accessibility_score(points_df, non_intersection_col_list, intersection_col_list)
```

This score is based on two list of column names: one for the columns to consider when the point is an intersection and one where it isn't. We calculate the highest ranks for each one of the features included in the score. We choose the highest one from among them (as we are considering that the higher the rank of each feature the worse the point is). Then we invert it to obtain the accessibility score. We make it so the scores range from 0 to 1. You can of course personalize the function as you like.

```python
def accessibility_score(points_df, non_intersection_col_list, intersection_col_list):

    obstacle_cols = non_intersection_col_list
    intersection_obstacle_cols = intersection_col_list


    for col in intersection_obstacle_cols:
        points_df[f'{col}_quantile'] = pd.qcut(points_df[col], 100, labels=np.arange(0,100)).astype(int)

    points_df['elevation_rank'] = points_df['elevation_rank'].astype(int)

    col_list = [f'{x}_quantile' for x in obstacle_cols] + ['elevation_rank']
    #col_list = ['elevation_rank']
    intersection_col_list = [f'{x}_quantile' for x in intersection_obstacle_cols] #+ ['elevation_rank']

    inaccessibility_scores = []

    for idx, point in points_df.iterrows():
        if point['is_intersection'] == False:
            #inaccessibility_scores.append(1)
            inaccessibility_scores.append(max(point[col_list]))

        else:
            inaccessibility_scores.append(max(point[intersection_col_list]))

    points_df['inaccessibility_score'] = inaccessibility_scores
    
    points_df['accessibility_score'] = (100 - points_df['inaccessibility_score'])/100

    return points_df
```
## Creating the city grid
To easily map a point to a score without having to always calculate the score for each new point, we create a city grid, which aggregates the information about the points in the surrounding area. This way, when we want information about any point in Lisbon, we give it the information connected to the closest grid center. To make the grid follow this code

```python
boundaries = {
    'north': 38.7957, # as tall as ericeira
    'south': 38.6914, # down to comporta
    'west': -9.229,
    'east': -9.0866 # as wide as montijo
}

grid = grid_maker(boundaries)
xx_centroids = grid[0].flatten()
yy_centroids = grid[1].flatten()

df = pd.DataFrame({'center_lat': yy_centroids, 'center_long': xx_centroids})
df['size'] = 1
df['is_land'] = df.apply(lambda x: globe.is_land(x['center_lat'], x['center_long']), axis=1)
df[['x_utm', 'y_utm']] = df.apply(lambda x: get_utm_grid(x), axis=1, result_type='expand')
```
This will essentially make a rectangular grid, but you will want to filter it according to the zone you are analyzing. In our case we only wanted the region of the municipality of Lisbon, so we filtered the other grid centers out, using a spatial join. You can run:

```python
filtered_grid = filter_grid(grid, address)
```
Now we can create the community score for each grid point. We will also use a distance-based decay, to make the points that are further away less relevant.

```python
community_score_grid = create_community_score_from_pois(grid, points_df, radius_in_meters)
```

## Routing

This also provides a routing functionality. We use the Open Route Service (ORS) API to map our route from point A to point B. These can be passed as coordinates or address names, although it is not guaranteed that ORS will identify the addresses.
To set up ORS, you first need to create an account. Once you have your API key, you can use the python SDK to make calls.

```python
client = ors.Client(key='your ORS API key')
```
To get the accessibility score of a given route, we will divide it into segments, and then into subsegments. Each subsegment will be associated with the score of the grid point closest to it. The segment score will be the average between all subsegment scores, and the same goes for the route's total accessibility score. We created Route and RouteSegment classes to be easier to handle all of these operations. You can see these classes in the [routing utils](path) script

```python

origin = "address_1" # you can also pass in the format "coords:lat,long"
destination = "address_2" # you can also pass in the format "coords:lat,long"

start_coords = parse(origin)
end_coords = parse(destination)

coordinates = [start_coords, end_coords]

client = ors.Client(key='your ORS API key')

route = client.directions(
    coordinates=coordinates,
    profile='foot-walking',
    alternative_routes = {"share_factor": 0.7,"target_count":3,"weight_factor":2},
    format='geojson',
    validate=False
)

geopoints = gpd.GeoDataFrame(full_grid['geos'],geometry='geos').rename(columns={'geos':'geometry'})
geopoints = geopoints.set_geometry('geometry')
pts3 = geopoints.geometry.unary_union
routes = [Route(x, full_grid, pts3) for x in route['features']]

scores = [route.score for route in routes]
main_route = routes[np.argmax(scores)]
alt_routes = [routes[idx] for idx, score in enumerate(scores) if idx != np.argmax(scores)]

```
## API
To run the API, we deployed a docker image into an AWS Lambda. The code for the API is in [this folder](api). To deploy your solution, you must run these two lines:
```bash
sam build
sam deploy --guided
```
For the API documentation, can refer to https://github.com/nilg-ai/voxpop_demo   

## Licenses
This code is released under CC BY-ND 4.0
This project is co-financed by the European Regional Development Fund through the Urban Innovative Actions Initiative

<img src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" data-canonical-src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" width="900" height="250" />


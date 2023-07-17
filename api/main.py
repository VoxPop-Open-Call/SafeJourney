from jinja2.runtime import V
from utils import ErrorCode, StatusCode, format_response, parse, Route, EnumWheelchair, map_score_to_color, not_null

from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from mangum import Mangum
import pandas as pd
import openrouteservice as ors
from shapely.geometry import Point
import geopandas as gpd
import numpy as np 
from shapely.ops import nearest_points

from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()

api_keys = [
    os.getenv("VOXPOP_API_KEY")
]  # This is encrypted in the database

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mangum allows us to use Lambdas to process requests
handler = Mangum(app=app)

from pymongo import MongoClient

client = MongoClient(
    os.getenv("MONGO_DB")
)

db = client.myFirstDatabase

"""
get_point(lat, long) ou get_point(id)

{'likes': x, 'dislikes': y, 'google_maps_url': z, 'features': [
    {'name': "High slope area", "icon": "s3.aws....", "probability": 0.3} ...
]
]

-----

"""
#full_grid = joblib.load('annotated_data/full_grid_totalscores.pkl')
retrieved_data = db['full_grid'].find({})
full_grid =pd.DataFrame(retrieved_data)
full_grid['geos'] = [Point(row['center_lat'], row['center_long']) for idx, row in full_grid.iterrows()]
geopoints = gpd.GeoDataFrame(full_grid['geos'],geometry='geos').rename(columns={'geos':'geometry'})
geopoints = geopoints.set_geometry('geometry')
geopoints['score'] = full_grid['accessibility_rank']
pts3 = geopoints.geometry.unary_union


@app.get("/get-point-score", status_code=status.HTTP_200_OK, dependencies=[Depends(api_key_auth)])
async def get_point_score(lat, longt):

# unary union of the gpd2 geomtries 
    def near(point, geopoints, pts=pts3):
        # find the nearest point and return the corresponding Place value
        nearest = geopoints[geopoints.geometry == nearest_points(point, pts)[1]]
        return nearest
    

    data = {
            'score': int(near(Point(float(lat), float(longt)), geopoints=geopoints).iloc[0]['score'])
    }

    print(data)
    return format_response(
        status=StatusCode.SUCCESS,
        message="Returned point metadata",
        message_code=ErrorCode.SUCCESS,
        data=[data]
    )


@app.get("/get-route", status_code=status.HTTP_200_OK, dependencies=[Depends(api_key_auth)])
async def get_route(origin: str, destination: str, wheelchair_type: EnumWheelchair):
    

    wheel_chair_type = wheelchair_type.value

    # TODO add SCORE, AVERAGE_ACCESSIBILITY, NEIGHBOURGOOD POINTS
    #address we need to geocode
    start_coords = parse(origin)
    end_coords = parse(destination)

    
    if start_coords is None or end_coords is None:
        return status.HTTP_404_NOT_FOUND
    #printing address and coordinates

    # Some coordinates in Lisbon
    coordinates = [start_coords, end_coords]

    client = ors.Client(key=os.getenv("ORS_CLIENT_KEY"))

    route = client.directions(
    coordinates=coordinates,
    profile='foot-walking',
    alternative_routes = {"share_factor": 0.7,"target_count":3,"weight_factor":2},
    format='geojson',
    #options={"avoid_features": ["steps"]},
    validate=False,
)

    routes = [Route(x, full_grid, pts3) for x in route['features']]

    scores = [route.score for route in routes]
    main_route = routes[np.argmax(scores)]
    alt_routes = [routes[idx] for idx, score in enumerate(scores) if idx != np.argmax(scores)]
    
    data = {'best_route': {
    'name': f'{main_route.segment_list[0].name} and {main_route.segment_list[-2].name}',
    'center': (main_route.get_street_linestring().centroid.x, main_route.get_street_linestring().centroid.y),
    'estimated_time': round(main_route.duration/60),
    'distance': round(main_route.distance, 2),
    'segments':[{
        'score': round((not_null(seg.score))/10, 2),
        'name': seg.name,
        'color': map_score_to_color(not_null(seg.score)/100),
        'subsegments': seg.subsegments,
        'origin': main_route.coordinates[seg.waypoints[0]],
        'destination': main_route.coordinates[seg.waypoints[1]],
        'instruction': seg.instruction,
        'distance': round(seg.distance, 2)

    } for seg in main_route.segment_list],
        'average_accessibility': round(main_route.score/10, 2),
        'color': map_score_to_color(main_route.score/100),
        'neighbourhood_points': []
    },
    'other_routes':[{
        'name': f'{alt_route.segment_list[0].name} and {alt_route.segment_list[-2].name}',
        'center': (alt_route.get_street_linestring().centroid.x, alt_route.get_street_linestring().centroid.y),
        'estimated_time': round(alt_route.duration/60),
        'distance': round(alt_route.distance, 2),
        'segments': [{
            'score': round(not_null(seg.score)/10, 2),
            'name': seg.name,
            'color': map_score_to_color(not_null(seg.score)/100),
            'subsegments': seg.subsegments,
            'origin': alt_route.coordinates[seg.waypoints[0]],
            'destination': alt_route.coordinates[seg.waypoints[1]],
            'instruction': seg.instruction,
            'distance': round(seg.distance, 2)
        } for seg in alt_route.segment_list],
        'average_accessibility': round(alt_route.score/10, 2),
        'color': map_score_to_color(alt_route.score/100),
        'neighbourhood_points': []

    } for alt_route in alt_routes]}

    return format_response(
            status=StatusCode.SUCCESS,
            message="Returned point metadata",
            message_code=ErrorCode.SUCCESS,
            data=[data]
        )


@app.get("/get-point-metadata", status_code=status.HTTP_200_OK, dependencies=[Depends(api_key_auth)])
async def get_point_metadata(lat: float, longt: float):
    """
    This endpoint will return the metadata for a point.
        {'likes': x, 'dislikes': y, 'google_maps_url': z, 'features': [
        {'name': "High slope area", "icon": "s3.aws....", "probability": 0.3} ...
    ]
    """

    # Fetch likes and dislikes from a database.
    # If point doesn't exist, insert to the database with 0 likes / dislikes.

    # Load the row from the predictions dataframe, to fetch google maps URL and the features list.
    # Map the features to an icon (python dictionary).
    try:
        point = db['new_point_sample'].find_one({'lat': lat, 'long': longt})
        #print(point)
            
    except:
        return status.HTTP_503_SERVICE_UNAVAILABLE


    mapping_class_url = {
        'bad_sidewalk_prob_quantile': 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/sidewalk_low_width.png',
        'bad_crosswalk_prob_quantile': 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/crosswalk_ramp.png',
        'car_quantile': 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/cars_sidewalk.png',
        'dist_to_cycle_quantile': 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/has_bikelane.png',
        'elevation_rank': 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/high_slope.png'
    }

    mapping_friendly_names = {
        'bad_sidewalk_prob_quantile': 'Accessible sidewalk',
        'bad_crosswalk_prob_quantile': 'Accessible crosswalk',
        'car_quantile': 'Without potential temporary obstacles',
        'dist_to_cycle_quantile': 'Proximity to bike lane',
        'elevation_rank': 'Flatness'
    }


    
    intersection_features = ['bad_sidewalk_prob_quantile', 'bad_crosswalk_prob_quantile', 'dist_to_cycle_quantile', 'elevation_rank', 'car_quantile']
    regular_features = ['bad_sidewalk_prob_quantile', 'dist_to_cycle_quantile', 'elevation_rank', 'car_quantile']



    try:

        if point['is_intersection']:
            present_features = intersection_features
        else:
            present_features = regular_features
        
        features = []

        for feature in present_features:
            
            if feature in mapping_class_url:
                icon = mapping_class_url[feature]
            else:
                icon = 'https://voxpop-icons.s3.eu-west-1.amazonaws.com/default.png'
            

            features.append(
                {
                    'label' : mapping_friendly_names[feature],
                    'prob': round((100 - point[feature]), 2),
                    'color': map_score_to_color(round((1 - point[feature]/100), 2)),
                    'icon': icon
                }
            )


    
    except Exception as e:
        print(e)
        return status.HTTP_404_NOT_FOUND


    data = {
        'likes': point['likes'],
        'dislikes': point['dislikes'],
        'lat': lat,
        'long': longt,
        'point_color_code': map_score_to_color(point['accessibility_score']),
        'GOOGLE_MAPS_URL': f'http://maps.google.com/maps?q=&layer=c&cbll={lat},{longt}&cbp=11,0,0,0,0',
        'address': point['street'],
        'features': features
        
    }


    return format_response(
            status=StatusCode.SUCCESS,
            message="Returned point metadata",
            message_code=ErrorCode.SUCCESS,
            data=[data],
        )


@app.get("/list-all-points", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(api_key_auth)])
async def list_all_points():
    """
    This endpoint will create the initial map.

    Potentially replace Nome_Rua by the coordinates.

    return [{'id': 1, 'title': 'Nome_Rua', 'latitude': x, 'longitude': y'}, ....}

    """
    # Get all data from database
    try:
        all_data = pd.DataFrame(db['new_point_sample'].find({}))
    except:
        return status.HTTP_503_SERVICE_UNAVAILABLE

    # Generate maps URL feature
    all_data = all_data.astype({"lat": 'str', "long": 'str'})
    all_data['GOOGLE_MAPS_URL'] = 'http://maps.google.com/maps?q=&layer=c&cbll='+all_data['lat']+','+all_data['long']+'&cbp=11,0,0,0,0' 


    # Delete unwanted features
    del all_data['_id']

    all_data['point_color_code'] = all_data['color']


    all_data = all_data.fillna('none')
    all_data['img_name'] = all_data['street']
    data_points = all_data.to_dict('records')

    return format_response(
            status=StatusCode.SUCCESS,
            message="Returned list of all points.",
            message_code=ErrorCode.SUCCESS,
            data=data_points,
        )



@app.put("/point-feedback", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(api_key_auth)])
async def give_point_feedback(lat: float, longt: float, is_like: bool = True, cookie: str = None):
    """
    This endpoint will return likes and dislikes for a given point.
    """
    # Default message 
    default_message = 'Your feedback has been registered!'
    already_voted_message = 'You already voted on this point!'
    no_cookies_message = 'You must accept cookies to provide feedback.'

    message = default_message
    status = StatusCode.SUCCESS
    message_code = ErrorCode.SUCCESS

    # Fetch row from database (where lat=lat and long = long)
    try:
        point = db['new_point_sample'].find_one({'lat': lat, 'long': longt})
    except:
        return status.HTTP_503_SERVICE_UNAVAILABLE

    try:
        nbr_likes = point['likes']
        nbr_dislikes = point['dislikes']
    except:
        return status.HTTP_404_NOT_FOUND

    if cookie:
        if db['session_logs'].find_one({'lat': lat, 'long': longt, 'cookie': cookie}) != None:
        
            message = already_voted_message
            status = StatusCode.ERROR
            message_code = ErrorCode.ERROR
        else:
            if is_like:
                nbr_likes = nbr_likes + 1
            else:
                nbr_dislikes = nbr_dislikes + 1

            # Update like number in database
            db['new_point_sample'].update_one(
            {'lat': lat, 'long':longt}, 
                    { 
                        "$set" :{ 'likes': nbr_likes, 'dislikes': nbr_dislikes}
                    }
            )

            db['session_logs'].insert_one(
                {'lat':lat, 'long': longt, 'cookie': cookie}
            )
    else:
        message = no_cookies_message
        status = StatusCode.ERROR
        message_code = ErrorCode.ERROR

    # Fetch upidated datapoint
    point = db['new_point_sample'].find_one({'lat': lat, 'long': longt})

    data = {
        'likes': point['likes'],
        'dislikes': point['dislikes'],
    }


    return format_response(
            status=status,
            message=message,
            message_code=message_code,
            data=[data],
        )
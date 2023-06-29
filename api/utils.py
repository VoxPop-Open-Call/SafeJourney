from enum import Enum
from geopy.geocoders import Nominatim
import folium
import shapely
import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Point

class RouteSegment():
    def __init__(self, step_dict, coord_list, grid, points):
        self.distance = step_dict['distance']
        self.duration = step_dict['duration']
        self.name = step_dict['name'] 
        self.instruction = step_dict['instruction']
        self.waypoints = step_dict['way_points']

        segment_coords = coord_list[self.waypoints[0] : (self.waypoints[1]+1)]

        self.subsegments = [{
            'origin': segment_coords[ix],
            'destination': segment_coords[ix+1]
        } for ix in range(len(segment_coords)-1)]

        def near(self,point, full_grid, pts):
        # find the nearest point and return the corresponding Place value
            nearest = full_grid[full_grid.geos == nearest_points(point, pts)[1]]
        
            return nearest

        def get_segment_score(self, grid, points):

            subseg_geos = [ Point(x['origin'][1], x['origin'][0]) for x in self.subsegments ]
            subseg_scores = [near(self,geo, grid, points)['accessibility_rank'] for geo in subseg_geos]

            if len(subseg_scores) > 0:
                return np.sum(subseg_scores)/len(subseg_scores)
                # return np.min(subseg_scores)
            else:
                return None

        self.score = get_segment_score(self, grid = grid, points = points)
        


class Route():
    def __init__(self, route_dict, grid, points):
        self.segment_list = [RouteSegment(x,route_dict['geometry']['coordinates'], grid, points) for x in route_dict['properties']['segments'][0]['steps']]
        self.distance = route_dict['properties']['summary']['distance']
        self.duration = route_dict['properties']['summary']['duration']
        self.coordinates = route_dict['geometry']['coordinates']

        self.score = sum(filter(None, [seg.score for seg in self.segment_list]))/len(self.segment_list)

    
    def get_street_linestring(self):
        polyline = folium.PolyLine(locations=[list(reversed(coord)) for coord in self.coordinates])

        return shapely.geometry.LineString(polyline.locations) 


class EnumWheelchair(str, Enum):
    electrical = "electrical"
    manual = "manual"


def map_score_to_color(score):

    color = '#353cdd' # blue

    if score != None:
        if score >= 0.8:
            color = '#72c472' # green
        elif score > 0.3:
            color= '#ffc700' # yellow
        else:
            color = '#ff6900' # orange
    
    return color


def parse(location):
    try:
        coords = [float(location.split(':')[1].split(',')[1]), float(location.split(':')[1].split(',')[0])]
    except Exception as e:
        geolocator = Nominatim(user_agent="my_request")
            
        #applying geocode method to get the location
        location = geolocator.geocode(location)
        coords = [location.longitude, location.latitude]

    return coords

def not_null(number):
    if number == None:
        return 0
    else:
        return number

class StatusCode(Enum):
    # This code is used when frontend call in backend is as expected.
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    REDIRECT = "REDIRECT"

class ErrorCode(object):
    """
    This class contains all the error codes which can be used further to send back as a response from a function.
    """

    SUCCESS = 0
    WARNING = 1
    ERROR = 2
    INCORRECT_FUNCTION_INPUT = 3
    ACTION_NOT_ALLOWED = 4

def format_response(
    status=StatusCode.SUCCESS, message="", message_code=ErrorCode.SUCCESS, data=[]
):
    """
    This is a helper function to format result of backend to send it to frontend.
    Args:
        status: (StatusCodeEnum object Must be of type ResultCode enum)
        message: Success or Error message (string, by default empty)
        data: data which needs to be send from the function(list)
    Returns:
        (Dict[str, Any): dict with following keys
            {
            "status":
            "message":
            "message_code":
            "data":
            }
    """

    output = {
        "STATUS": status.value,
        "MESSAGE": message,
        "MESSAGE_CODE": message_code,
        "DATA": data,
    }

    return output #json.loads(json_util.dumps(output, default=str))
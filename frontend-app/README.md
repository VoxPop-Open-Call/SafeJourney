## Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).
The project node 18.14.0

## Project Structure

```
src
├── assets # Contains static assets like images, fonts, etc.
├── components # Reusable React components
├── interfaces # TypeScript interfaces or types for data structures
├── utils # Utility functions or helper modules
├── App.tsx # Entry point of your React app
└── index.tsx # Main entry point for rendering the React app
```

## Description

Specialized navigation application designed to provide directions for individuals with mobility challenges, like using an wheelchair. It aims to offer an accessible and user-friendly mapping service. The app leverages an API to retrieve directions from a starting point to a destination, taking into account various accessibility factors.
Key Features

-   **Wheelchair-friendly Routing:** The app calculates and displays routes that are accessible to individuals using wheelchairs, considering factors such as the existence of sidewalk / crosswalk, the existence of a ramp in crosswalks, sidewalk width and sidewalk pavement type.
-   **Scoring System:** The API used by the app provides paths with scores, indicating the level of accessibility for each route. The app prioritizes the best score to recommend the optimal route for the user.
-   **Point of Interest Markers:** The map view displays markers representing various points of interest for individuals in wheelchairs. Each marker provides important accessibility information such as "Irregular Sidewalk," "Crosswalk Contains a Ramp," or "No Crosswalk."
-   **User Validation:** Users can view the details of each point of interest and contribute to the validation process by upvoting or downvoting the accuracy of the provided information.
-   **Point-to-Point Navigation:** Users can enter their starting point and destination, and the app generates step-by-step directions to guide them along the most accessible path
-   **User-Friendly Interface:** The app features a user-friendly interface designed to be accessible to individuals with varying levels of mobility and technological proficiency.

The Accessible Directions App strives to empower individuals with mobility challenges, specifically those using wheelchairs, to navigate their surroundings with confidence. By promoting inclusivity and accessibility in daily life, the app aims to enhance the quality of life for wheelchair users by providing reliable, accurate, and personalized navigation assistance.

## To run the project

To run the project locally you simple should run
`npm install`
and then
`npm start`

## Usage

-   **Loading Points of Interest:**
    -   Upon launching the app, the user is presented with a map view that displays important points for individuals in wheelchairs.
    -   Markers on the map represent various locations, each accompanied by relevant information such as "Irregular Sidewalk", "Crosswalk Contains a Ramp" or "No Crosswalk."
    -   Each point of interest is associated with a score that reflects the level of accessibility, indicating the quality of the provided information.
-   **Reviewing Point Details:**
    -   Interested users can select a marker on the map to view its detailed information.
    -   The point details display the specific accessibility features, and the associated score for that information.
    -   Users have the option to upvote or downvote the accuracy of the information, contributing to the community-driven validation process.
-   **Getting Directions:**
    -   To get directions, the user has multiple options:
        -   Clicking the "Directions" button within the point details view.
        -   Clicking the "Directions" button within the marker popup on the map.
        -   Right-clicking on the map and selecting "Directions to Here" or "Directions from Here."
    -   The user populates both the origin and destination fields to define their desired route.
    -   After submitting the request, the app generates multiple route options based on accessibility.
    -   Each route is accompanied by a score reflecting its level of wheelchair accessibility.
    -   The user can explore the details of each route to understand the specific accessibility features and decide which one suits their needs best.

By providing comprehensive point information, allowing users to review and validate the details, and offering multiple options for obtaining accessible directions.

## API Calls

### Get Point of Interest

**Endpoint:** GET `/list-all-points`

**Description:** This API call retrieves the list of points of interest markers for individuals in wheelchairs. It returns an array of markers representing various locations on the map

**Response:**

```
{
  MESSAGE: "Returned list of all points."
  MESSAGE_CODE: 0,
  STATUS: "SUCCESS",
  DATA: [{
    "lat": "38.7565536",
    "long": "-9.1399819",
    "pavement_type": "irregular",
    "crosswalk_level": "ramp",
    "sidewalk_width": "sufficient",
    "sidewalk_level": "none",
    "address": "Av. Rio de Janeiro 54C, 1700-125 Lisboa, Portugal",
    "likes": 0,
    "dislikes": 0,
    "slope": 0,
    "GOOGLE_MAPS_URL": "http://maps.google.com/maps?q=&layer=c&cbll=38.7565536,-9.1399819&cbp=11,0,0,0,0",
    "point_color_code": "#00FF00",
    "img_name": "Av. Rio de Janeiro 54C, 1700-125 Lisboa, Portugal"
  }]
}
```

### Get Marker Details

**Endpoint:** GET `/get-point-metadata?lat=${lat}&longt=${long}`

**Description:** This API call retrieves detailed information about a specific point of interest marker. The `lat` and `longt` parameters in the endpoint represent the latitude and longitude coordinates of the marker. The response includes information such as the accessibility features and the associated score provided by other users. Users can access this information to gain a better understanding of the accessibility of a particular location.

**Response:**

```
{
  MESSAGE_CODE: 0,
  STATUS: "SUCCESS",
  DATA: {
    "likes": 0,
    "dislikes": 0,
    "lat": 38.78397169209848,
    "long": -9.12971420489252,
    "point_color_code": "#00FF00",
    "GOOGLE_MAPS_URL": "http://maps.google.com/maps?q=&layer=c&cbll=38.78397169209848,-9.12971420489252&cbp=11,0,0,0,0",
    "address": "Estrada da Circunvalação 12, 2685 Prior Velho, Portugal",
    "features": [
      {
        "label": "Irregular sidewalk",
        "prob": 0.81273973,
        "icon": "https://voxpop-icons.s3.eu-west-1.amazonaws.com/irregular_sidewalk.png"
      },
      {
        "label": "Crosswalk with sufficient width",
        "prob": 0.77279687,
        "icon": "https://voxpop-icons.s3.eu-west-1.amazonaws.com/default.png"
      },
      {
        "label": "No crosswalk",
        "prob": 0.56207496,
        "icon": "https://voxpop-icons.s3.eu-west-1.amazonaws.com/default.png"
      }
    ]
  }
}
```

### Upvote/Downvote Marker Information

**Endpoint:** PUT `/point-feedback?lat=${lat}&longt=${long}&is_like=${like}`

**Description:** This API call allows users to upvote or downvote the accuracy of the information associated with a specific marker. The `lat` and `longt` parameters in the endpoint represent the latitude and longitude coordinates of the marker. The `is_like` parameter is set to true for an upvote and false for a downvote
**Response:**

```
{
  "MESSAGE": "Your feedback has been registered!",
  "DATA": [
    {
      "likes": 1,
      "dislikes": 0
    }
  ]
}
```

### Get Directions

**Endpoint:** `GET /get-route?origin=${originValue}&destination=${destinationValue}&wheelchair_type=${selectedType}`

**Description:** This API call retrieves directions from a specified origin to a destination. The `origin` and `destination` parameters represent the starting and ending locations respectively. The wheelchair_type parameter indicates the type of wheelchair being used, allowing the app to consider specific accessibility requirements. The API calculates and returns multiple route options based on accessibility factors, along with associated scores indicating the level of wheelchair accessibility for each route.

**Response:**

```
{
  "STATUS": "SUCCESS",
  "MESSAGE": "Returned point metadata",
  "MESSAGE_CODE": 0,
  "DATA": [
    {
      "best_route": {
        "name": "Rua Augusta and Rua Salgueiro Maia",
        "center": [
            38.748224155458296,
            -9.131696371714233
        ],
        "estimated_time": 119.63000000000001,
        "distance": 9969.3,
        "segments": [
          {
            "score": 0.9,
            "color": "#72c472",
            "origin": [
                -9.137112,
                38.709223
            ],
            "destination": [
                -9.137533,
                38.710279
            ],
            "instruction": "Head north on Rua Augusta",
            "distance": 123,
            "subsegments": [
              {
                  "origin": [
                      -9.101019,
                      38.761548
                  ],
                  "destination": [
                      -9.100978,
                      38.76176
                  ]
              },
              {
                  "origin": [
                      -9.100978,
                      38.76176
                  ],
                  "destination": [
                      -9.100972,
                      38.761792
                  ]
              }
            ]
          },
          {
            "score": 0.9,
            "color": "#72c472",
            "origin": [
                -9.12971,
                38.783987
            ],
            "destination": [
                -9.12971,
                38.783987
            ],
            "instruction": "Arrive at Rua Salgueiro Maia, on the left",
            "distance": 0,
            "subsegments": []
          }
        ],
        "average_accessibility": 0.9,
        "color": "#72c472",
        "neighbourhood_points": []
      },
      "other_routes": [
        {
          "name": "Rua de São Julião and Rua Salgueiro Maia",
          "center": [
            38.74714396030597,
            -9.13073950898695
          ],
          "estimated_time": 122.31,
          "distance": 10192.6,
          "segments": [
            {
              "score": 0.5,
              "color": "#ffc700",
              "origin": [
                -9.137112,
                38.709223
              ],
              "destination": [
                -9.134505,
                38.709855
              ],
              "instruction": "Head east on Rua de São Julião",
              "distance": 236.9,
              "subsegments": []
            },
            {
              "score": 0.5,
              "color": "#ffc700",
              "origin": [
                -9.12971,
                38.783987
              ],
              "destination": [
                -9.12971,
                38.783987
              ],
              "instruction": "Arrive at Rua Salgueiro Maia, on the left",
              "distance": 0,
              "subsegments": []
          }
          ],
          "average_accessibility": 0.5,
          "color": "#ffc700",
          "neighbourhood_points": []
        }
      ]
    }
  ]
}
```

## Licenses

This code is released under CC BY-ND 4.0

This project is co-financed by the European Regional Development Fund through the Urban Innovative Actions Initiative

<img src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" data-canonical-src="https://github.com/nilg-ai/voxpop-nilgai/blob/dev/figures/all_logos.png" width="900" height="250" />

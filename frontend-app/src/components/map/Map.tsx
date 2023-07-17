import { useEffect, useState } from 'react'

import L, { LatLng, Map as LeafletMap } from 'leaflet'
import {
    TileLayer,
    MapContainer,
    Popup,
    useMapEvents,
    Polyline,
} from 'react-leaflet'
import { IMarker } from '../../interfaces/IMarker'
import Spinner from '../Spinner'
import { IRoute } from '../../interfaces/IRoute'
import LocationMarkers from './LocationMarkers'
import DirectionsPopup from './DirectionsPopup'

import 'leaflet/dist/leaflet.css'
import './Map.scss'

function Map({
    selectMarker,
    setOrigin,
    setDestination,
    routeClick,
    selectedRoute,
    directionRoutes,
}: {
    selectMarker: (marker: IMarker) => void
    setOrigin: (origin: string) => void
    setDestination: (destination: string) => void
    routeClick: (route: IRoute) => void
    selectedRoute: IRoute | null
    directionRoutes: IRoute[]
}) {
    const [isLoading, setLoading] = useState<boolean>(false)
    const [showDirectionsPopup, setShowDirectionsPopup] = useState(false)
    const [cursorPos, setCursorPos] = useState<LatLng | null>(null)
    const [map, setMap] = useState<LeafletMap | null>(null)

    useEffect(() => {
        if (map && directionRoutes.length > 0) {
            const bestRoute = directionRoutes[0]
            const point = bestRoute.center
            const bounds = new L.LatLng(point[0], point[1]).toBounds(
                bestRoute.distance
            )
            map.flyToBounds(bounds)
        }
    }, [directionRoutes, map])

    const MapEvents = () => {
        useMapEvents({
            contextmenu: (event) => {
                setCursorPos(event.latlng)
                setShowDirectionsPopup(true)
            },
            popupclose: () => {
                setCursorPos(null)
            },
        })
        return <></>
    }

    return (
        <div className="relative">
            {isLoading ? (
                <div className="absolute z-20 flex h-full w-full items-center justify-center bg-slate-400 opacity-70">
                    <Spinner />
                </div>
            ) : (
                <></>
            )}
            <MapContainer
                className="relative z-10 h-screen"
                center={[38.75, -9.15]}
                zoom={13}
                ref={setMap}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationMarkers
                    selectMarker={selectMarker}
                    setDestination={setDestination}
                    setLoading={setLoading}
                    isLoading={isLoading}
                ></LocationMarkers>

                <MapEvents />

                {showDirectionsPopup && cursorPos && (
                    <Popup position={cursorPos}>
                        <DirectionsPopup
                            position={cursorPos}
                            setOrigin={setOrigin}
                            setDestination={setDestination}
                        />
                    </Popup>
                )}

                {selectedRoute
                    ? selectedRoute.segments.map((point) =>
                          point.subsegments.map((subPoint, k) => (
                              <Polyline
                                  key={k}
                                  positions={[
                                      [subPoint.origin[1], subPoint.origin[0]],
                                      [
                                          subPoint.destination[1],
                                          subPoint.destination[0],
                                      ],
                                  ]}
                                  color={point.color}
                                  weight={8}
                                  smoothFactor={1}
                              />
                          ))
                      )
                    : directionRoutes.map((route, i) => {
                          return route.segments.map((point) =>
                              point.subsegments.map((subPoint, k) => (
                                  <Polyline
                                      key={k}
                                      positions={[
                                          [
                                              subPoint.origin[1],
                                              subPoint.origin[0],
                                          ],
                                          [
                                              subPoint.destination[1],
                                              subPoint.destination[0],
                                          ],
                                      ]}
                                      color={
                                          i === directionRoutes.length - 1
                                              ? point.color
                                              : '#8E8EA6'
                                      }
                                      weight={8}
                                      smoothFactor={1}
                                      eventHandlers={{
                                          click: () => routeClick(route),
                                      }}
                                  />
                              ))
                          )
                      })}
            </MapContainer>
        </div>
    )
}

export default Map

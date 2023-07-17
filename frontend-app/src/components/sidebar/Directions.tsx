import { useEffect, useState } from 'react'
import { BsThreeDots } from 'react-icons/bs'
import { FaDotCircle, FaFlagCheckered } from 'react-icons/fa'
import { HiOutlineArrowsUpDown } from 'react-icons/hi2'

import { IRoute } from '../../interfaces/IRoute'
import AvailableRoutes from './AvailableRoutes'
import RouteDetails from './RouteDetails'
import Spinner from '../Spinner'
import useDebounce from '../../utils/use-debounce'

const types = [
    { id: 1, name: 'Manual', type: 'manual' },
    { id: 2, name: 'Electrical', type: 'electrical' },
]

function Directions({
    origin,
    destination,
    setDirectionRoutes,
    selectRoute,
    selectedRoute,
}: {
    origin: string
    destination: string
    setDirectionRoutes: (routes: IRoute[]) => void
    selectRoute: (route: IRoute | null) => void
    selectedRoute: IRoute | null
}) {
    const [selectedType, setSelectedType] = useState<string>(types[0].type)
    const [routes, setRoutes] = useState<IRoute[]>([])

    const [originValue, setOriginValue] = useState(origin)
    const [destinationValue, setDestinationValue] = useState(destination)

    const [isLoading, setLoading] = useState<boolean>(false)
    const [routeNotFound, setRouteNotFound] = useState<boolean>(false)

    const debouncedRequest = useDebounce(() => {
        if (originValue && destinationValue && selectedType) {
            setLoading(true)
            setRouteNotFound(false)
            setRoutes([])
            setDirectionRoutes([])
            selectRoute(null)

            fetch(
                `${process.env.REACT_APP_API_URL}/get-route?origin=${originValue}&destination=${destinationValue}&wheelchair_type=${selectedType}`,
                {
                    headers: {
                        'Content-Type': 'application/json',
                        Authorization: `Bearer ${process.env.REACT_APP_BEARER_TOKEN}`,
                    },
                }
            )
                .then((res) => res.json())
                .then((res) => {
                    if (res.STATUS === 'SUCCESS') {
                        // for showing the route details
                        setRoutes([
                            res.DATA[0].best_route,
                            ...res.DATA[0].other_routes,
                        ])
                        // in the map the last route is on top
                        setDirectionRoutes([
                            ...res.DATA[0].other_routes,
                            res.DATA[0].best_route,
                        ])
                    } else {
                        setRouteNotFound(true)
                    }
                    setLoading(false)
                })
                .catch((err) => {
                    console.error(err)
                    setRouteNotFound(true)
                    setLoading(false)
                })
        }
    }, 2000)

    useEffect(() => {
        setOriginValue(origin)
        debouncedRequest()
    }, [origin, debouncedRequest])

    useEffect(() => {
        setDestinationValue(destination)
        debouncedRequest()
    }, [destination, debouncedRequest])

    function onSwitchClick() {
        const localOrigin = `${originValue}`
        const localDestination = `${destinationValue}`
        setOriginValue(localDestination)
        setDestinationValue(localOrigin)
        debouncedRequest()
    }

    function onRouteClick(route: IRoute) {
        selectRoute(route)
    }

    function onBackClick() {
        selectRoute(null)
    }

    return (
        <section>
            <div className="grid grid-cols-[50px_minmax(0,_1fr)_50px] items-center">
                <div className="flex flex-col items-center gap-5">
                    <FaDotCircle className="text-nilg-blue" />
                    <div className="h-4 border-2 border-dashed border-nilg-gray"></div>
                    <FaFlagCheckered className="text-nilg-orange" />
                </div>
                <div className="flex flex-col gap-4">
                    <input
                        className="block w-full rounded-[8px] p-3 text-gray-900 ring-1 ring-nilg-gray focus:outline-none"
                        type="text"
                        value={originValue}
                        onChange={(e) => {
                            setOriginValue(e.target.value)
                            debouncedRequest()
                        }}
                    />
                    <input
                        className="block w-full rounded-[8px] p-3 text-gray-900 ring-1 ring-nilg-gray focus:outline-none"
                        type="text"
                        value={destinationValue}
                        onChange={(e) => {
                            setDestinationValue(e.target.value)
                            debouncedRequest()
                        }}
                    />
                </div>
                <div className="flex flex-col items-center gap-12">
                    <BsThreeDots />
                    <button onClick={() => onSwitchClick()}>
                        <HiOutlineArrowsUpDown />
                    </button>
                </div>
            </div>

            <div className="mt-4 flex items-center border-t border-nilg-gray px-5 py-3">
                <span className="mr-4 text-sm font-semibold text-nilg-dark-gray">
                    Wheelchair type:
                </span>
                <div>
                    <select
                        id="type"
                        name="type"
                        className="block w-full rounded-md border-0 py-1.5 text-nilg-black"
                        defaultValue={types[0].type}
                        onChange={(e) => {
                            setSelectedType(e.target.value)
                            debouncedRequest()
                        }}
                    >
                        {types.map((type) => (
                            <option key={type.id} value={type.type}>
                                {type.name}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {isLoading && (
                <div className="absolute top-0 z-20 flex h-full w-full items-center justify-center bg-slate-400 opacity-70">
                    <Spinner />
                </div>
            )}

            {routeNotFound && routes.length === 0 && (
                <div className="p-5">Route not found</div>
            )}

            {routes.length > 0 ? (
                selectedRoute ? (
                    <RouteDetails
                        route={selectedRoute}
                        backPressed={onBackClick}
                    />
                ) : (
                    <AvailableRoutes
                        routes={routes}
                        routeClick={onRouteClick}
                    />
                )
            ) : (
                <></>
            )}
        </section>
    )
}

export default Directions

import { AiOutlineClose } from 'react-icons/ai'
import { IMarker } from '../../interfaces/IMarker'
import MarkerDetail from './MarkerDetail'
import Directions from './Directions'
import { IRoute } from '../../interfaces/IRoute'

function Sidebar({
    selectedMarker,
    closeDetails,
    origin,
    destination,
    selectedRoute,
    setDestination,
    selectRoute,
    setDirectionRoutes,
}: {
    selectedMarker: IMarker | null
    closeDetails: () => void
    origin: string
    destination: string
    selectedRoute: IRoute | null
    setDestination: (destination: string) => void
    selectRoute: (route: IRoute | null) => void
    setDirectionRoutes: (routes: IRoute[]) => void
}) {
    return (
        <div className="flex h-full flex-col overflow-auto bg-white shadow-[0_0_12px_rgb(0,0,0,0.1)]">
            <div className="absolute right-0 top-0">
                <button
                    className="bg-transparent p-4 text-black hover:bg-transparent focus:ring-0 focus:ring-transparent active:bg-transparent"
                    onClick={closeDetails}
                >
                    <AiOutlineClose />
                </button>
            </div>
            {selectedMarker ? (
                <MarkerDetail
                    selectedMarker={selectedMarker}
                    setDestination={setDestination}
                />
            ) : (
                <div className="mt-10">
                    <Directions
                        origin={origin}
                        destination={destination}
                        selectRoute={selectRoute}
                        selectedRoute={selectedRoute}
                        setDirectionRoutes={setDirectionRoutes}
                    ></Directions>
                </div>
            )}
        </div>
    )
}

export default Sidebar

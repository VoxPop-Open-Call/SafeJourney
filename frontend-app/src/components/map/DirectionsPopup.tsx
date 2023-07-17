import { LatLng } from 'leaflet'

function DirectionsPopup({
    position,
    setOrigin,
    setDestination,
}: {
    position: LatLng
    setOrigin: (origin: string) => void
    setDestination: (destination: string) => void
}) {
    return (
        <div className="flex flex-col">
            <button
                onClick={() =>
                    setOrigin(`coords:${position.lat},${position.lng}`)
                }
                className="rounded-t-lg p-2 hover:bg-slate-100"
            >
                Directions from here
            </button>

            <button
                onClick={() =>
                    setDestination(`coords:${position.lat},${position.lng}`)
                }
                className="rounded-b-lg p-2 hover:bg-slate-100"
            >
                Directions to here
            </button>
        </div>
    )
}

export default DirectionsPopup

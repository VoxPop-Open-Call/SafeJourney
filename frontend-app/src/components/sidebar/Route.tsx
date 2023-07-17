import { IRoute } from '../../interfaces/IRoute'
import ScorePill from './ScorePill'

function Route({
    route,
    routeClick,
}: {
    route: IRoute
    routeClick: () => void
}) {
    return (
        <section
            className="flex cursor-pointer cursor-pointer items-center gap-4 border-b border-nilg-gray bg-white p-5"
            onClick={routeClick}
        >
            <div className="text-sm font-semibold text-nilg-black">
                {route.name}
            </div>
            <div className="ml-auto flex flex-col whitespace-nowrap text-right">
                <div className="text-sm font-semibold text-nilg-black">
                    {route.estimated_time} min
                </div>
                <div className="text-xs font-semibold text-nilg-dark-gray">
                    {route.distance} m
                </div>
            </div>
            <ScorePill route={route} />
        </section>
    )
}

export default Route

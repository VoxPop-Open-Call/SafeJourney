import { IRoute } from '../../interfaces/IRoute'
import Route from './Route'

function AvailableRoutes({
    routes,
    routeClick,
}: {
    routes: IRoute[]
    routeClick: (route: IRoute) => void
}) {
    return (
        <section className="my-5 bg-nilg-gray">
            <div className="bg-white px-5 text-xs font-semibold text-nilg-dark-gray">
                Recommended Route
            </div>
            <Route route={routes[0]} routeClick={() => routeClick(routes[0])} />

            <div className="mt-2.5 bg-white px-5 pt-5 text-xs font-semibold text-nilg-dark-gray">
                Other Routes
            </div>
            {routes.slice(1).map((route, i) => (
                <Route
                    key={i}
                    route={route}
                    routeClick={() => routeClick(route)}
                />
            ))}
        </section>
    )
}

export default AvailableRoutes

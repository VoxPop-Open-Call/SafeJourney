import { IoArrowBackOutline } from 'react-icons/io5'

import { IRoute } from '../../interfaces/IRoute'
import ScorePill from './ScorePill'

function Start({
    route,
    backPressed,
}: {
    route: IRoute
    backPressed: () => void
}) {
    return (
        <section className="flex cursor-pointer cursor-pointer items-center gap-2 border-nilg-gray bg-white px-5 pb-5">
            <button
                className="border-1 flex items-center gap-2 rounded-[40px] border border-nilg-gray px-4 py-2.5 text-sm font-semibold"
                onClick={() => backPressed()}
            >
                <IoArrowBackOutline className="text-black" />
                <span className="nilg-black">Go back</span>
            </button>
            <div className="ml-auto mr-3 flex flex-col whitespace-nowrap text-right">
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

export default Start

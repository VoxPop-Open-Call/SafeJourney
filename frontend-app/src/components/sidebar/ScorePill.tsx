import { AiFillStar } from 'react-icons/ai'
import { IRoute } from '../../interfaces/IRoute'
import { useEffect } from 'react'

function ScorePill({ route }: { route: IRoute }) {
    useEffect(() => {
        console.log(route)
    }, [route])
    return (
        <div
            className="justify-content flex items-center gap-1 rounded-md px-2 py-1 text-sm font-semibold text-white"
            style={{ backgroundColor: route.color }}
        >
            <AiFillStar /> {route.average_accessibility}
        </div>
    )
}

export default ScorePill

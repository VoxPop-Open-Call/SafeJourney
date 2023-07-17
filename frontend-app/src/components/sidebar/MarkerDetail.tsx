import { useEffect, useState } from 'react'
import { IMarker } from '../../interfaces/IMarker'
import { FaThumbsUp, FaThumbsDown, FaRegMap } from 'react-icons/fa'
import { BsArrow90DegRight } from 'react-icons/bs'
import Feature from '../feature/Feature'
import getCookie from '../../utils/get-cookie'
import Spinner from '../Spinner'

const MarkerDetail = ({
    selectedMarker,
    setDestination,
}: {
    selectedMarker: IMarker | null
    setDestination: (destination: string) => void
}) => {
    const [selectedMarkerDetail, setMarkerDetail] = useState<
        IMarker | undefined
    >(undefined)
    const [isLoading, setLoading] = useState<boolean>(false)
    const [isLikeLoading, setLikeLoading] = useState<boolean>(false)
    const [alreadyVoted, setVote] = useState<boolean>(false)
    const [message, setMessage] = useState<string>()

    useEffect(() => {
        if (selectedMarker) {
            getMarkerDetails(selectedMarker.lat, selectedMarker.long)
        }
    }, [selectedMarker])

    const getMarkerDetails = async (lat: number, long: number) => {
        setLoading(true)
        setVote(false)

        fetch(
            `${process.env.REACT_APP_API_URL}/get-point-metadata?lat=${lat}&longt=${long}`,
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
                    const marker = res.DATA[0]
                    setMarkerDetail({
                        lat: marker.lat,
                        long: marker.long,
                        likes: marker.likes,
                        dislikes: marker.dislikes,
                        directionsUrl: marker.GOOGLE_MAPS_URL,
                        features: marker.features.map((f: any) => ({
                            label: f.label,
                            prob: f.prob,
                            icon: f.icon,
                            color: f.color,
                        })),
                        address: marker.address,
                    })
                }
                setLoading(false)
            })
            .catch((err) => {
                setLoading(false)
                console.error(err)
            })
    }

    const onAction = (lat?: number, long?: number, like: boolean = true) => {
        setLikeLoading(true)
        const cookie: string | undefined = getCookie('userId')
        fetch(
            `${
                process.env.REACT_APP_API_URL
            }/point-feedback?lat=${lat}&longt=${long}&is_like=${like}&cookie=${
                cookie ? cookie : ''
            }`,
            {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${process.env.REACT_APP_BEARER_TOKEN}`,
                },
                body: JSON.stringify({}),
            }
        )
            .then((response) => response.json())
            .then((res) => {
                if (res.STATUS === 'SUCCESS') {
                    const marker = res.DATA[0]
                    setMarkerDetail({
                        lat: marker.lat,
                        long: marker.long,
                        likes: marker.likes,
                        dislikes: marker.dislikes,
                        directionsUrl: marker.GOOGLE_MAPS_URL,
                        features: selectedMarkerDetail?.features?.map(
                            (f: any) => ({
                                label: f.label,
                                prob: f.prob,
                                icon: f.icon,
                                color: f.color,
                            })
                        ),
                        address: selectedMarkerDetail?.address,
                    })
                }
                setMessage(res.MESSAGE)
                setVote(true)
                setLikeLoading(false)
            })
            .catch((error) => {
                setLikeLoading(false)
                console.error('Error:', error)
            })
    }

    return (
        <>
            {isLoading ? (
                <div className="mt-3 self-center">
                    <Spinner />
                </div>
            ) : (
                <>
                    <div className="mt-5 p-3 text-base font-medium">
                        {selectedMarkerDetail?.address}
                    </div>
                    <div className="flex items-center p-3 text-slate-300">
                        <FaThumbsUp />{' '}
                        <span className="ml-1">
                            {selectedMarkerDetail?.likes}
                        </span>
                        <FaThumbsDown className="ml-5" />{' '}
                        <span className="ml-1">
                            {selectedMarkerDetail?.dislikes}
                        </span>
                    </div>
                    <hr className="my-3" />
                    <div className="mt-3 flex justify-center gap-10">
                        <div className="flex flex-col items-center">
                            <button
                                className="flex h-14 w-14 items-center justify-center rounded-full bg-nilg-blue"
                                onClick={() =>
                                    setDestination(
                                        `coords:${selectedMarker?.lat},${selectedMarker?.long}`
                                    )
                                }
                            >
                                <BsArrow90DegRight className="text-2xl font-semibold text-white" />
                            </button>
                            <div className="mt-1 text-xs font-semibold">
                                Directions
                            </div>
                        </div>
                        <div className="flex flex-col items-center">
                            <a
                                href={selectedMarkerDetail?.directionsUrl}
                                target="_blank"
                                rel="noreferrer"
                            >
                                <button className="flex h-14 w-14 items-center justify-center rounded-full border border-nilg-gray bg-white">
                                    <FaRegMap className="text-2xl font-semibold !text-nilg-blue" />
                                </button>
                            </a>
                            <div className="mt-1 text-xs font-semibold">
                                View Region
                            </div>
                        </div>
                    </div>
                    <hr className="my-3" />
                    <div className="grid grid-cols-3 p-3">
                        {selectedMarkerDetail?.features?.map((f, i) => (
                            <Feature
                                key={i}
                                label={f.label}
                                prob={f.prob}
                                icon={f.icon}
                                color={f.color}
                            />
                        ))}
                    </div>
                    {alreadyVoted ? (
                        <div className="mt-auto flex flex-col items-center justify-center bg-nilg-blue p-3 text-white">
                            <div className="text-center text-2xl font-bold">
                                {message}
                            </div>
                        </div>
                    ) : isLikeLoading ? (
                        <div className="mt-auto self-center p-3">
                            <Spinner />
                        </div>
                    ) : (
                        <div className="mt-auto mt-auto flex flex-col items-center justify-center bg-nilg-blue p-3 text-white">
                            <div className="text-center text-2xl font-bold">
                                Do you agree with the classification?
                            </div>
                            <div className="mt-3 flex gap-3">
                                <div className="text-center">
                                    <button
                                        onClick={() =>
                                            onAction(
                                                selectedMarkerDetail?.lat,
                                                selectedMarkerDetail?.long
                                            )
                                        }
                                        className="flex h-20 w-20 items-center justify-center rounded-full bg-white"
                                    >
                                        <FaThumbsUp className="text-3xl text-nilg-green" />
                                    </button>
                                    <div className="mt-1 text-base">Yes</div>
                                </div>
                                <div className="text-center">
                                    <button
                                        onClick={() =>
                                            onAction(
                                                selectedMarkerDetail?.lat,
                                                selectedMarkerDetail?.long,
                                                false
                                            )
                                        }
                                        className="flex h-20 w-20 items-center justify-center rounded-full bg-white"
                                    >
                                        <FaThumbsDown className="text-3xl text-nilg-orange" />
                                    </button>
                                    <div className="mt-1 text-base">No</div>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}
        </>
    )
}

export default MarkerDetail

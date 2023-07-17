function Feature({
    label,
    prob,
    icon,
    color,
}: {
    label: string
    prob: number
    icon: string
    color: string
}) {
    return (
        <div className="mb-3 flex flex-col items-center text-center">
            <div className="rounded-full border-solid border-slate-500">
                <img src={icon} alt={label} />
            </div>
            <div className="text-xs font-medium">{label}</div>
            <div style={{ color }} className={`text-xs font-semibold`}>
                {prob}
            </div>
        </div>
    )
}

export default Feature

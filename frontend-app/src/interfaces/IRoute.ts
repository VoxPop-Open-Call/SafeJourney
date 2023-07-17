export interface IRoute {
    id: number
    name: string
    center: [number, number]
    distance: number
    estimated_time: number
    segments: {
        instruction: string
        color: string
        origin: [number, number]
        destination: [number, number]
        score: number
        distance: number
        subsegments: {
            origin: [number, number]
            destination: [number, number]
        }[]
    }[]
    average_accessibility: number
    color: string
}

import debounce from 'lodash-es/debounce'
import { useEffect, useMemo, useRef, useCallback } from 'react'

const useDebounce = (
    callback: () => void,
    debounceTime: number = 1000
): (() => void) => {
    const ref = useRef<() => void>()

    useEffect(() => {
        ref.current = callback
    }, [callback])

    const debouncedCallback = useMemo(() => {
        const func = () => {
            ref.current?.()
        }

        return debounce(func, debounceTime)
    }, [debounceTime])

    return useCallback(() => {
        debouncedCallback()
    }, [debouncedCallback])
}

export default useDebounce

import { useState } from 'react'
import { FaSearch } from 'react-icons/fa'
import { HiArrowUturnRight } from 'react-icons/hi2'

function SearchInput({
    searchChange,
}: {
    searchChange: (value: string) => void
}) {
    const [searchValue, setSearchValue] = useState<string>()

    return (
        <form
            className="relative"
            onSubmit={(e) => {
                e.preventDefault()
                searchValue && searchChange(searchValue)
            }}
        >
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <FaSearch
                    className="h-5 w-5 text-gray-400"
                    aria-hidden="true"
                />
            </div>
            <input
                type="text"
                name="search"
                id="search"
                className="md:w-[335px] block w-full rounded-[44px] px-10 py-3 text-base text-gray-900 ring-1 ring-nilg-gray focus:outline-none"
                placeholder="Search"
                onChange={(e) => {
                    setSearchValue(e.target.value)
                }}
            />
            <button
                type="submit"
                className="absolute right-3 top-3 h-5 w-5 text-gray-400"
            >
                <HiArrowUturnRight className="text-2xl font-semibold" />
            </button>
        </form>
    )
}

export default SearchInput

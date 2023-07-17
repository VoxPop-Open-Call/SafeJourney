import { useState } from 'react'
import SearchInput from './SearchInput'
import Brand from './Brand'
import { Dialog } from '@headlessui/react'
import { FaBars } from 'react-icons/fa'
import { AiOutlineClose } from 'react-icons/ai'

function Header({searchChange}: {searchChange: (value: string) => void}) {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

    return (
        <header className="bg-white">
            <nav
                className="mx-auto flex items-center justify-between py-3 px-6"
                aria-label="Global"
            >
                <Brand />
                <div className="flex sm:hidden">
                    <button
                        type="button"
                        className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
                        onClick={() => setMobileMenuOpen(true)}
                    >
                        <span className="sr-only">Open main menu</span>
                        <FaBars className="h-6 w-6" aria-hidden="true" />
                    </button>
                </div>
                <div className="ml-6 hidden sm:flex sm:gap-x-12">
                    <SearchInput searchChange={searchChange}/>
                </div>
                <div className="hidden sm:flex sm:flex-1 sm:justify-end">
                    <a
                        href="https://nilg.ai/safejourney/"
                        type="button"
                        target="_blank"
                        rel="noreferrer"
                    >
                        <button className="focus:nilg-black text-sm text-nilg-black">
                            About
                        </button>
                    </a>
                </div>
            </nav>
            <Dialog
                as="div"
                className="sm:hidden"
                open={mobileMenuOpen}
                onClose={setMobileMenuOpen}
            >
                <Dialog.Panel className="fixed inset-y-0 right-0 z-40 w-full overflow-y-auto bg-white py-3 px-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10 h-[70vh]">
                    <div className="flex items-center justify-between">
                        <Brand />
                        <button
                            type="button"
                            className="-m-2.5 rounded-md p-2.5 text-gray-700"
                            onClick={() => setMobileMenuOpen(false)}
                        >
                            <span className="sr-only">Close menu</span>
                            <AiOutlineClose
                                className="h-6 w-6"
                                aria-hidden="true"
                            />
                        </button>
                    </div>
                    <div className="mt-6 flow-root">
                        <div className="divide-y divide-gray-500/10">
                            <SearchInput searchChange={searchChange}/>
                        </div>
                    </div>
                    <div className="mt-6 flow-root">
                        <div className="-my-6 divide-y divide-gray-500/10">
                            <div className="py-3">
                                <a
                                    href="https://nilg.ai/safejourney/"
                                    className="-mx-3 block rounded-lg px-3 py-2.5 font-semibold leading-7 text-nilg-black text-gray-900 hover:bg-gray-50 focus:text-nilg-black"
                                >
                                    About
                                </a>
                            </div>
                        </div>
                    </div>
                </Dialog.Panel>
            </Dialog>
        </header>
    )
}

export default Header

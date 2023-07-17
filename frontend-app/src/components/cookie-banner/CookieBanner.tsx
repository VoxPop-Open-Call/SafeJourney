import { Dialog, Transition } from '@headlessui/react'
import { Fragment, useState } from 'react'
import { AiOutlineClose } from 'react-icons/ai'
import { v4 as uuid } from 'uuid'

function CookieModal({
    show,
    onDismiss,
    onAccept,
}: {
    show: boolean
    onDismiss: (state: boolean) => void
    onAccept: () => void
}) {
    return (
        <Transition.Root show={show} as={Fragment}>
            <Dialog
                as="div"
                className="relative z-10"
                onClose={() => onDismiss(false)}
            >
                <Transition.Child
                    as={Fragment}
                    enter="ease-out duration-300"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in duration-200"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                >
                    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
                </Transition.Child>

                <div className="fixed inset-0 z-10 overflow-y-auto">
                    <div className="flex min-h-full justify-center pt-12 text-center md:items-center md:p-4">
                        <Transition.Child
                            as={Fragment}
                            enter="ease-out duration-300"
                            enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                            enterTo="opacity-100 translate-y-0 sm:scale-100"
                            leave="ease-in duration-200"
                            leaveFrom="opacity-100 translate-y-0 sm:scale-100"
                            leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                        >
                            <Dialog.Panel className="relative mx-6 h-[530px] transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all md:mx-32 md:h-full">
                                <div className="absolute right-0 top-0 block pr-5 pt-5">
                                    <button
                                        type="button"
                                        className="rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none"
                                        onClick={() => onDismiss(false)}
                                    >
                                        <span className="sr-only">Close</span>
                                        <AiOutlineClose
                                            className="h-6 w-6"
                                            aria-hidden="true"
                                        />
                                    </button>
                                </div>
                                <div className="border-b p-5">
                                    <h1 className="text-xl font-medium text-gray-900">
                                        Privacy Overview
                                    </h1>
                                </div>

                                <div className="h-96 space-y-6 overflow-auto p-5 text-base leading-relaxed text-gray-500 md:h-full">
                                    <p>
                                        This website uses cookies to improve
                                        your experience while you navigate
                                        through the website. Out of these, the
                                        cookies that are categorized as
                                        necessary are stored on your browser as
                                        they are essential for the working of
                                        basic functionalities of the website. We
                                        also use third-party cookies that help
                                        us analyze and understand how you use
                                        this website. These cookies will be
                                        stored in your browser only with your
                                        consent. You also have the option to
                                        opt-out of these cookies. But opting out
                                        of some of these cookies may affect your
                                        browsing experience.
                                    </p>
                                    <p>
                                        Check our cookie policy, terms, and
                                        privacy policy
                                    </p>
                                    <ul>
                                        <li>
                                            <a
                                                className="font-medium text-nilg-blue hover:underline"
                                                href="https://nilg.ai/terms-of-service/"
                                                target="_blank"
                                                rel="noreferrer"
                                            >
                                                https://nilg.ai/terms-of-service/
                                            </a>
                                        </li>
                                        <li>
                                            <a
                                                className="font-medium text-nilg-blue hover:underline"
                                                href="https://nilg.ai/privacy-policy/"
                                                target="_blank"
                                                rel="noreferrer"
                                            >
                                                https://nilg.ai/privacy-policy/
                                            </a>
                                        </li>
                                        <li>
                                            <a
                                                className="font-medium text-nilg-blue hover:underline"
                                                href="https://nilg.ai/cookie-policy/"
                                                target="_blank"
                                                rel="noreferrer"
                                            >
                                                https://nilg.ai/cookie-policy/
                                            </a>
                                        </li>
                                    </ul>
                                </div>

                                <div className="flex w-full justify-end border-t p-5">
                                    <button
                                        className="focus:!ring-2 group flex h-min items-center justify-center rounded-lg border border-transparent bg-green-700 p-0.5 text-center font-medium text-white hover:bg-green-800 focus:z-10 focus:ring-4 focus:ring-green-300 disabled:hover:bg-green-700 "
                                        onClick={() => onAccept()}
                                    >
                                        <span className="flex items-center rounded-md px-4 py-2 text-sm">
                                            I accept
                                        </span>
                                    </button>
                                </div>
                            </Dialog.Panel>
                        </Transition.Child>
                    </div>
                </div>
            </Dialog>
        </Transition.Root>
    )
}

function CookieBanner() {
    const [visible, setVisible] = useState(false)
    const [acceptedCookie, setCookie] = useState(
        document.cookie != null && document.cookie !== ''
    )

    function acceptCookies(): void {
        document.cookie = `userId=${uuid()}`
        setCookie(true)
    }

    return (
        <>
            {!acceptedCookie ? (
                <>
                    <div className="fixed bottom-0 left-0 z-50 flex w-screen items-center justify-center bg-white">
                        <div className="w-9/12 p-3">
                            We use cookies on our website to give you the most
                            relevant experience by remembering your preferences
                            and repeat visits. By clicking “Accept All”, you
                            consent to the use of ALL the cookies. However, you
                            may visit "Cookie Settings" to provide a controlled
                            consent
                        </div>
                        <div className="flex w-3/12 flex-col justify-center gap-2 p-3 md:flex-row">
                            <button
                                className="focus:!ring-2 group flex h-min items-center justify-center rounded-lg border border-gray-300 bg-white p-0.5 text-center font-medium text-gray-900 hover:bg-gray-100 focus:z-10 focus:ring-4 focus:ring-blue-300 disabled:hover:bg-white"
                                color="light"
                                onClick={() => setVisible(true)}
                            >
                                <span className="flex items-center rounded-md px-4 py-2 text-sm">
                                    Cookie Settings
                                </span>
                            </button>

                            <button
                                className="focus:!ring-2 group flex h-min items-center justify-center rounded-lg border border-transparent bg-green-700 p-0.5 text-center font-medium text-white hover:bg-green-800 focus:z-10 focus:ring-4 focus:ring-green-300 disabled:hover:bg-green-700"
                                onClick={() => acceptCookies()}
                            >
                                <span className="flex items-center rounded-md px-4 py-2 text-sm">
                                    Accept All
                                </span>
                            </button>
                        </div>
                    </div>
                    <CookieModal
                        show={visible}
                        onDismiss={setVisible}
                        onAccept={acceptCookies}
                    />
                </>
            ) : (
                <></>
            )}
        </>
    )
}

export default CookieBanner

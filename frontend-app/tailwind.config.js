/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ['./src/**/*.{js,jsx,ts,tsx}'],
    theme: {
        extend: {
            colors: {
                'nilg-blue': '#353cdd',
                'nilg-green': '#72c472',
                'nilg-yellow': '#ffc700',
                'nilg-orange': '#ff6900',
                'nilg-gray': '#E6E6F2',
                'nilg-dark-gray': '#8E8EA6',
                'nilg-black': '#24272F',
            },
        },
    },
    safelist: [
        {
            pattern: /(bg|text|border)-nilg-(blue|orange|yellow|green)/,
        },
    ],
}

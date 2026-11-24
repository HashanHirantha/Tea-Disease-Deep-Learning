/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tea: {
          50: '#f0f9f3',
          100: '#dbf2e2',
          200: '#b9e5c7',
          300: '#89d1a3',
          400: '#54b67b',
          500: '#329a5c',
          600: '#237c48',
          700: '#1e623b',
          800: '#1b4e31',
          900: '#174029',
          950: '#0b2416',
        },
      },
    },
  },
  plugins: [],
};

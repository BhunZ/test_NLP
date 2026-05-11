import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: '#0b0d14',
        'bg-elevated': '#14171f',
        'bg-elevated-2': '#1c2030',
        border: '#262b3d',
        'border-subtle': '#1d2130',
        text: '#f0f2f8',
        'text-dim': '#9aa0b4',
        'text-muted': '#6b7088',
        accent: {
          DEFAULT: '#8b5cf6',
          hover: '#7c3aed',
          soft: '#2d2547',
        },
        success: '#34d399',
        warning: '#fbbf24',
        danger: '#f87171',
      },
      fontFamily: {
        tight: ['Inter Tight', 'sans-serif'],
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        lg: '16px',
        md: '12px',
        sm: '8px',
      },
      boxShadow: {
        sm: '0 1px 2px rgba(0,0,0,0.2)',
        DEFAULT: '0 4px 12px rgba(0,0,0,0.3)',
        lg: '0 12px 32px rgba(0,0,0,0.4)',
        glow: '0 0 0 3px rgba(139,92,246,0.15)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} satisfies Config;

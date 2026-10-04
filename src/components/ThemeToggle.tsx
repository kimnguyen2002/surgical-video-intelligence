import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../services/theme';

export function ThemeToggle({ className = '' }: { className?: string }) {
  const [theme, toggle] = useTheme();
  const next = theme === 'dark' ? 'light' : 'dark';
  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className={`flex h-8 w-8 items-center justify-center rounded-md border border-line bg-card text-fg2 transition-colors hover:bg-hover hover:text-fg ${className}`}
    >
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

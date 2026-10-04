import React, { Suspense, lazy, useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import About from './pages/About';

const App = lazy(() => import('./App'));

export type Route = 'home' | 'console';

/**
 * Routing, in the smallest form that works everywhere this app runs.
 *
 * The console is served from a static bundle and is also embedded in a
 * cross-origin iframe on AI Studio, where a History-API router would need the
 * host to serve a rewrite for every path. The hash needs nothing: `#/console` is
 * the console, anything else is the home page, and a plain `<a href="#/console">`
 * navigates without a library.
 *
 * The home page is what people land on, so it stays in the main chunk and the
 * console, which carries the inference runtime, is split out.
 */
function readRoute(): Route {
  const hash = window.location.hash.replace(/^#\/?/, '').split(/[?#]/)[0].toLowerCase();
  return hash === 'console' ? 'console' : 'home';
}

const PageLoader = () => (
  <div className="min-h-screen bg-page flex items-center justify-center">
    <Loader2 className="w-5 h-5 text-accent animate-spin" />
  </div>
);

export default function Root() {
  const [route, setRoute] = useState<Route>(readRoute);

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // A route change is a new page, so it starts at the top rather than wherever
  // the previous one happened to be scrolled to.
  useEffect(() => {
    window.scrollTo(0, 0);
    document.documentElement.setAttribute('data-surface', route === 'console' ? 'console' : 'home');
  }, [route]);

  if (route === 'home') return <About />;

  return (
    <Suspense fallback={<PageLoader />}>
      <App />
    </Suspense>
  );
}

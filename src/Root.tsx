import React, { Suspense, lazy, useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import App from './App';

const About = lazy(() => import('./pages/About'));

export type Route = 'console' | 'about';

/**
 * Routing, in the smallest form that works everywhere this app runs.
 *
 * The console is served from a static bundle and is also embedded in a
 * cross-origin iframe on AI Studio, where a History-API router would need the
 * host to serve a rewrite for every path. The hash needs nothing: `#/about` is
 * a sub page, anything else is the console, and a plain `<a href="#/about">`
 * navigates without a library.
 *
 * The console stays in the main chunk — it is the page people land on, and a
 * loading flash there would be a regression. The About page is split out.
 */
function readRoute(): Route {
  const hash = window.location.hash.replace(/^#\/?/, '').split(/[?#]/)[0].toLowerCase();
  return hash === 'about' ? 'about' : 'console';
}

const PageLoader = () => (
  <div className="min-h-screen bg-[#070a0f] flex items-center justify-center">
    <Loader2 className="w-5 h-5 text-cyan-500 animate-spin" />
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
  }, [route]);

  if (route === 'about') {
    return (
      <Suspense fallback={<PageLoader />}>
        <About />
      </Suspense>
    );
  }

  return <App />;
}

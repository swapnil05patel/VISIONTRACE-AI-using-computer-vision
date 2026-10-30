import { ClerkProvider, Show, SignIn, useAuth } from '@clerk/react';
import { useEffect, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { setAuthTokenGetter, setDevUserId } from '@workspace/api-client-react';

import App from './App';
import { ErrorBoundary } from '@/components/error-boundary';

import './index.css';

const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || import.meta.env.CLERK_PUBLISHABLE_KEY;
const devUser = import.meta.env.VITE_VISIONTRACE_DEV_USER_ID as string | undefined;

function ClerkTokenBridge() {
  const { getToken } = useAuth();
  useEffect(() => {
    setAuthTokenGetter(async () => (await getToken()) ?? null);
    return () => setAuthTokenGetter(null);
  }, [getToken]);
  return null;
}

function Root({ children }: { children: ReactNode }) {
  if (!publishableKey) {
    if (devUser) {
      setDevUserId(devUser);
    }
    return <>{children}</>;
  }
  return (
    <ClerkProvider publishableKey={publishableKey}>
      <Show when="signed-out">
        <div className="app-shell" style={{ placeItems: 'center', minHeight: '100vh' }}>
          <SignIn />
        </div>
      </Show>
      <Show when="signed-in">
        <ClerkTokenBridge />
        {children}
      </Show>
    </ClerkProvider>
  );
}

createRoot(document.getElementById('root')!, {
  onCaughtError: (error, errorInfo) => {
    console.error(error, errorInfo.componentStack);
  },
}).render(
  <ErrorBoundary>
    <Root>
      <App />
    </Root>
  </ErrorBoundary>,
);

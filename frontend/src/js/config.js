/**
 * TruthLens API Configuration Resolver
 *
 * Dynamically resolves the backend API Base URL across:
 * 1. Vite build-time environment variable: VITE_API_BASE_URL (configured in Vercel)
 * 2. Window runtime configuration: window.__TRUTHLENS_CONFIG__.API_BASE_URL
 * 3. Same-origin fallback (empty string) for local dev or monolithic proxies
 */

export const getApiBaseUrl = () => {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
      return import.meta.env.VITE_API_BASE_URL.trim().replace(/\/+$/, '');
    }
  } catch (e) {
    // Fallback if import.meta is not evaluated
  }

  if (typeof window !== 'undefined' && window.__TRUTHLENS_CONFIG__?.API_BASE_URL) {
    return window.__TRUTHLENS_CONFIG__.API_BASE_URL.trim().replace(/\/+$/, '');
  }

  return '';
};

export const API_BASE_URL = getApiBaseUrl();

import axios from 'axios';

const api = axios.create({
  baseURL: '/api', // Proxy in Vite handles the rest
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptors if needed
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export default api;

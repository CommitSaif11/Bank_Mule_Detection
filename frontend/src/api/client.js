import axios from 'axios'

const API = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000' })

export const getStats = () => API.get('/api/stats')
export const getAlerts = (params) => API.get('/api/alerts', { params })
export const getAccounts = (params) => API.get('/api/accounts', { params })
export const getAccount = (id) => API.get(`/api/accounts/${id}`)
export const getExplain = (id) => API.get(`/api/explain/${id}`)

export const getUploadStatus = () => API.get('/api/upload/status')
export const uploadDataset = (formData) =>
  API.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })

export const getLiveSample = () => API.get('/api/score/live/sample')
export const postLiveScore = (payload) => API.post('/api/score/live', payload)

export default API

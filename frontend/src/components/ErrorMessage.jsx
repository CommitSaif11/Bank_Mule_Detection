import { AlertTriangle } from 'lucide-react'

export default function ErrorMessage({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <AlertTriangle className="text-red-400 mb-3" size={36} />
      <p className="text-gray-700 dark:text-gray-200 font-medium">Unable to reach the API</p>
      <p className="text-gray-400 text-sm mt-1">{message || 'Please make sure the backend server is running on localhost:8000'}</p>
    </div>
  )
}

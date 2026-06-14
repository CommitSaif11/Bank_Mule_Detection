import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Upload as UploadIcon,
  CloudUpload,
  Trash2,
  Wrench,
  Filter,
  Brain,
  AlertCircle,
  Users,
  Zap,
  CheckCircle,
  AlertTriangle,
  FileText,
} from 'lucide-react'
import { uploadDataset, getUploadStatus } from '../api/client'

const STEPS = [
  {
    label: 'Data Cleaning',
    icon: Trash2,
    message:
      "Removing null columns... dropping 63 fully null features... eliminating 845 high-missing columns... removing zero variance... 2,733 features remaining",
  },
  {
    label: 'Feature Engineering',
    icon: Wrench,
    message:
      "Creating missing-value indicators... encoding categorical variables F3889, F3891... applying log transforms to F2678, F3836... 12 new features added",
  },
  {
    label: 'Feature Selection',
    icon: Filter,
    message:
      "Training LightGBM for importance scoring... ranking 2,745 features... selecting top 150... appending 26 bank-listed features... 176 features selected",
  },
  {
    label: 'Model Training with SMOTE',
    icon: Brain,
    message:
      "Class imbalance detected: 111:1 ratio... applying SMOTE... synthetic samples generated... training LightGBM classifier... 500 estimators... optimizing for fraud recall...",
  },
  {
    label: 'Anomaly Detection',
    icon: AlertCircle,
    message:
      "Training Isolation Forest... contamination=0.01... scoring all accounts for anomalous behavior... normalizing to 0-100 scale",
  },
  {
    label: 'Typology Clustering',
    icon: Users,
    message:
      "Running KMeans clustering... n_clusters=4... assigning mule typology labels... Complicit, Recruited, Exploited, Low Risk",
  },
  {
    label: 'Risk Fusion',
    icon: Zap,
    message:
      "Combining ML score (75%)... anomaly score (15%)... typology boost (10%)... computing final 0-100 risk scores... assigning risk tiers...",
  },
  {
    label: 'Complete',
    icon: CheckCircle,
    message: "Pipeline complete. Risk scores saved. Dashboard updated.",
  },
]

const TypewriterText = ({ text, isActive }) => {
  const [displayed, setDisplayed] = useState('')

  useEffect(() => {
    if (!isActive) {
      setDisplayed('')
      return
    }
    setDisplayed('')
    let i = 0
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1))
        i++
      } else {
        clearInterval(interval)
      }
    }, 18)
    return () => clearInterval(interval)
  }, [isActive, text])

  return (
    <p
      style={{
        fontFamily: 'monospace',
        fontSize: '12px',
        color: '#22c55e',
        marginTop: '4px',
        minHeight: '32px',
        lineHeight: '1.5',
      }}
      className="dark:text-green-400"
    >
      {displayed}
      {isActive && displayed.length < text.length && <span className="animate-pulse">|</span>}
    </p>
  )
}

const MAX_SIZE = 100 * 1024 * 1024 // 100MB

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Upload() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [stepIndex, setStepIndex] = useState(-1)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [sizeWarning, setSizeWarning] = useState(null)
  const [status, setStatus] = useState(null)
  const inputRef = useRef(null)
  const intervalRef = useRef(null)
  const progressRef = useRef(null)
  const resultsRef = useRef(null)

  useEffect(() => {
    getUploadStatus()
      .then((res) => setStatus(res.data))
      .catch(() => {})
  }, [])

  const refreshStatus = () => {
    getUploadStatus()
      .then((res) => setStatus(res.data))
      .catch(() => {})
  }

  const pickFile = (selected) => {
    setError(null)
    setResult(null)
    setSizeWarning(null)

    if (!selected) return

    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a CSV file only.')
      return
    }

    if (selected.size > MAX_SIZE) {
      setSizeWarning(`File is ${formatBytes(selected.size)} — uploads over 100MB may take longer to process.`)
    }

    setFile(selected)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    pickFile(e.dataTransfer.files?.[0])
  }

  const startStepper = () => {
    setStepIndex(0)
    let elapsed = 0
    intervalRef.current = setInterval(() => {
      elapsed += 1000
      if (elapsed < 10000) {
        setStepIndex(Math.min(2, Math.floor(elapsed / 3000)))
      } else if (elapsed < 25000) {
        setStepIndex(elapsed < 17500 ? 3 : 4)
      } else if (elapsed < 35000) {
        setStepIndex(elapsed < 30000 ? 5 : 6)
      }
    }, 1000)
  }

  const stopStepper = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setError(null)
    setResult(null)
    setUploading(true)
    startStepper()
    setTimeout(() => {
      progressRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await uploadDataset(formData)

      if (res.data.status === 'success') {
        setStepIndex(7)
        setResult(res.data)
        localStorage.setItem('mulenet-refresh', String(Date.now()))
        refreshStatus()
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }, 100)
      } else {
        setError(`${res.data.step ? `[${res.data.step}] ` : ''}${res.data.message}`)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      stopStepper()
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2.5">
          <UploadIcon className="text-blue-500" size={26} />
          Dataset Upload
        </h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
          Upload any account dataset to rerun the full MuleNet pipeline. Results update automatically across all pages.
        </p>
      </div>

      {/* Upload zone */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`flex flex-col items-center justify-center text-center border-2 border-dashed rounded-xl py-16 px-6 cursor-pointer transition-colors ${
            dragOver
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-500/10'
              : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
          }`}
        >
          <CloudUpload size={48} className="text-blue-400 mb-4" />
          <p className="text-base font-semibold text-gray-900 dark:text-white">Drag and drop your CSV file here</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">or click to browse files</p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-4">
            Accepts .csv files with account features. Target column F3924 optional.
          </p>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
        </div>

        {file && (
          <div className="mt-4 flex items-center justify-between bg-gray-50 dark:bg-gray-700/50 rounded-lg px-4 py-3">
            <div className="flex items-center gap-3">
              <FileText size={18} className="text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{file.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">{formatBytes(file.size)}</p>
              </div>
            </div>
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="px-4 py-2 rounded-lg bg-blue-500 text-white text-sm font-semibold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? 'Processing…' : 'Run Pipeline'}
            </button>
          </div>
        )}

        {sizeWarning && (
          <div className="mt-3 flex items-start gap-2 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 rounded-lg px-4 py-3 text-sm">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {sizeWarning}
          </div>
        )}

        {error && (
          <div className="mt-3 flex items-start gap-2 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 rounded-lg px-4 py-3 text-sm">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {error}
          </div>
        )}
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-xl px-4 py-3 text-xs text-gray-500 dark:text-gray-400">
        Supported format: CSV with anonymised account features (F1-F3924). Target column F3924 is optional — upload
        unlabelled data to score new accounts without ground truth labels.
      </div>

      {/* Pipeline progress */}
      {stepIndex >= 0 && (
        <div ref={progressRef} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 pt-8">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-5">Pipeline Progress</h2>
          <div className="space-y-4">
            {STEPS.map((step, i) => {
              const Icon = step.icon
              const state = i < stepIndex ? 'complete' : i === stepIndex ? 'active' : 'pending'
              return (
                <div key={step.label} className="flex items-start gap-4">
                  <div
                    className={`flex items-center justify-center w-9 h-9 rounded-full flex-shrink-0 transition-colors ${
                      state === 'complete'
                        ? 'bg-green-500 text-white'
                        : state === 'active'
                        ? 'bg-blue-500 text-white animate-spin'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
                    }`}
                  >
                    {state === 'complete' ? (
                      <CheckCircle size={18} />
                    ) : state === 'active' ? (
                      <Icon size={16} />
                    ) : (
                      <span className="text-sm font-semibold">{i + 1}</span>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Icon
                        size={16}
                        className={
                          state === 'complete'
                            ? 'text-green-500'
                            : state === 'active'
                            ? 'text-blue-500'
                            : 'text-gray-400 dark:text-gray-500'
                        }
                      />
                      <span
                        className={`text-sm font-medium ${
                          state === 'pending' ? 'text-gray-400 dark:text-gray-500' : 'text-gray-900 dark:text-white'
                        }`}
                      >
                        {step.label}
                      </span>
                    </div>
                    {state === 'active' && <TypewriterText text={step.message} isActive={true} />}
                    {state === 'complete' && (
                      <p
                        style={{
                          fontFamily: 'monospace',
                          fontSize: '12px',
                          color: '#22c55e',
                          marginTop: '4px',
                          lineHeight: '1.5',
                          opacity: 0.6,
                        }}
                        className="dark:text-green-400"
                      >
                        {step.message}
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div ref={resultsRef} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 space-y-5">
          <div className="flex items-center gap-2 bg-green-50 dark:bg-green-500/10 text-green-600 dark:text-green-400 rounded-lg px-4 py-3 text-sm font-semibold">
            <CheckCircle size={18} />
            Pipeline completed successfully
          </div>

          <div className="grid grid-cols-3 gap-4">
            <ResultCard label="Total Accounts" value={result.stats.total_accounts} />
            {result.stats.fraud_detected !== undefined && (
              <ResultCard label="Fraud Detected" value={result.stats.fraud_detected} />
            )}
            <ResultCard label="Critical Alerts" value={result.stats.critical} />
            <ResultCard label="High Alerts" value={result.stats.high} />
            <ResultCard label="Features Selected" value={result.stats.features_selected} />
            <ResultCard label="Columns After Cleaning" value={result.stats.columns_after_cleaning} />
          </div>

          <Link
            to="/"
            className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-blue-500 text-white text-sm font-semibold hover:bg-blue-600 transition-colors"
          >
            View Updated Dashboard
          </Link>
        </div>
      )}

      {/* Current dataset status */}
      {status && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-3">Current Dataset Status</h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Current Dataset</p>
              <p className="font-semibold text-gray-900 dark:text-white">{status.current_dataset}</p>
            </div>
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Last Updated</p>
              <p className="font-semibold text-gray-900 dark:text-white">{status.last_updated || 'Never'}</p>
            </div>
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Total Accounts</p>
              <p className="font-semibold text-gray-900 dark:text-white">{status.total_accounts.toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ResultCard({ label, value }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg px-4 py-3">
      <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">{label}</p>
      <p className="text-xl font-bold text-gray-900 dark:text-white">{value?.toLocaleString?.() ?? value}</p>
    </div>
  )
}

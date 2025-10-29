import React, { useState, useCallback } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, Video, FileText, RefreshCw, CheckCircle, XCircle, Clock, Settings, Play, Zap, Menu, X, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import SubtitleSyncAPI from './api';
import SubtitleGeneratorAPI from './generatorApi';
import FileManagerSidebar from './components/FileManagerSidebar';
import SubtitlePreviewPage from './pages/SubtitlePreviewPage';
import SubtitleGenerator from './components/SubtitleGenerator';
import VideoPlayer from './components/VideoPlayer';
import SettingsPanel from './components/SettingsPanel';

// placeholder import for login
// import Login from './LoginPlaceholder'; // placeholder import for login (simulated)
const Login = () => <div>Login Placeholder</div>;

const api = new SubtitleSyncAPI();
const generatorApi = new SubtitleGeneratorAPI();

// TitleBar Component
const TitleBar = ({ onToggleSidebar, sidebarOpen = false, onOpenSettings }) => {
  return (
    <div className="title-bar">
      <div className="title-bar-container">
        {/* Logo and Name Section */}
        <div className="title-bar-left">
          {/* Logo */}
          <div className="title-bar-logo">
            <span>🎬</span>
          </div>
          {/* App Name */}
          <div className="title-bar-text">
            <h1>Subtitle Tools</h1>
            <p>Generate and sync subtitles</p>
          </div>
        </div>

        {/* Sidebar and Settings Toggle Buttons */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            className="title-bar-button"
            onClick={onToggleSidebar}
            aria-label="Toggle sidebar"
          >
            {/* Icon changes based on sidebar state  */}
            {sidebarOpen ? (
              <X size={20} />
            ) : (
              <Menu size={20} />
            )}
            {/* Button text - hidden on small screens  */}
            <span className="button-text-short">
              {sidebarOpen ? 'Close' : ''}
            </span>
            {/* Full text for larger screens  */}
            <span className="button-text-full">
              File Manager
            </span>
          </button>
          {/* Settings Toggle Button */}
          <button
            className="title-bar-button"
            onClick={onOpenSettings}
            aria-label="Toggle settings"
          >
            <Settings size={20} />
            <span className="button-text-short"></span>
            <span className="button-text-full">Settings</span>
          </button>
        </div>
      </div>
    </div>
  );
};

// --- QualityCheckOptions component for collapsible details ---
const QualityCheckOptions = ({
  syncChecked, setSyncChecked,
  overlapChecked, setOverlapChecked
}) => {
  const [showSyncDetails, setShowSyncDetails] = useState(false);
  const [showOverlapDetails, setShowOverlapDetails] = useState(false);

  return (
    <div
      className="quality-check-container"
      style={{
        marginTop: '20px',
        marginBottom: '10px',
        padding: '24px',
        background: 'rgba(255,255,255,0.95)',
        borderRadius: '12px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
      }}
    >
      <h3 style={{
        color: '#333',
        marginBottom: '24px',
        fontWeight: 600,
        fontSize: '1.25em',
        textAlign: 'left',
      }}>
        Quality Check Options
      </h3>

      <div style={{ 
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
        alignItems: 'start'
      }}>
        {/* Alignment & Completeness Card */}
        <CheckOptionCard
          id="sync"
          checked={syncChecked}
          setChecked={setSyncChecked}
          showDetails={showSyncDetails}
          setShowDetails={setShowSyncDetails}
          title="Alignment & Completeness"
          description="Fix sync errors and subtitle gaps"
          color="#4CAF50"
          icon={<CheckCircle size={20} />}
          details={[
            'Audio-subtitle synchronization',
            'Identify incorrect or missing subtitles',
            'Fix subtitle timing gaps'
          ]}
        />

        {/* Overlapping Issues Card */}
        <CheckOptionCard
          id="overlap"
          checked={overlapChecked}
          setChecked={setOverlapChecked}
          showDetails={showOverlapDetails}
          setShowDetails={setShowOverlapDetails}
          title="Overlapping Issues"
          description="Reposition subtitles for overlap"
          color="#2196F3"
          icon={<AlertCircle size={20} />}
          details={[
            'Detect overlapping subtitle lines',
            'Reposition subtitles to avoid burnt-in text',
            'Improve subtitle readability'
          ]}
        />
      </div>
    </div>
  );
};

// Separate card component for better organization
const CheckOptionCard = ({ 
  id, checked, setChecked, showDetails, setShowDetails, 
  title, description, color, icon, details 
}) => {
  const baseColor = color;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        background: `${baseColor}0d`,
        borderRadius: '12px',
        border: `2px solid ${checked ? baseColor : '#e0e0e0'}`,
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        boxShadow: checked ? `0 4px 12px ${baseColor}30` : '0 2px 8px rgba(0,0,0,0.06)`
      }}
    >
      {/* Main clickable area for checkbox */}
      <div
        onClick={() => setChecked(!checked)}
        onKeyPress={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setChecked(!checked);
          }
        }}
        role="button"
        tabIndex={0}
        aria-pressed={checked}
        style={{
          padding: '18px 20px',
          cursor: 'pointer',
          background: checked ? `${baseColor}08` : 'transparent',
          transition: 'background 0.2s ease',
          outline: 'none'
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = checked ? `${baseColor}12` : `${baseColor}08`}
        onMouseLeave={(e) => e.currentTarget.style.background = checked ? `${baseColor}08` : 'transparent'}
        onFocus={(e) => e.currentTarget.style.outline = `3px solid ${baseColor}40`}
        onBlur={(e) => e.currentTarget.style.outline = 'none'}
      >
        <div style={{ 
          display: 'flex', 
          alignItems: 'flex-start',
          gap: '14px'
        }}>
          {/* Custom Checkbox */}
          <div
            style={{
              minWidth: '24px',
              width: '24px',
              height: '24px',
              borderRadius: '6px',
              border: `2px solid ${checked ? baseColor : '#d0d0d0'}`,
              background: checked ? baseColor : 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
              marginTop: '2px'
            }}
          >
            {checked && (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path 
                  d="M2 7L5.5 10.5L12 3" 
                  stroke="white" 
                  strokeWidth="2.5" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </div>

          {/* Content */}
          <div style={{ flex: 1 }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center',
              gap: '8px',
              marginBottom: '6px'
            }}>
              <span style={{ color: baseColor, display: 'flex' }}>
                {icon}
              </span>
              <h4 style={{
                margin: 0,
                fontWeight: 600,
                color: '#333',
                fontSize: '1.05em'
              }}>
                {title}
              </h4>
            </div>
            <p style={{
              margin: 0,
              color: '#666',
              fontSize: '0.9em',
              lineHeight: '1.5'
            }}>
              {description}
            </p>
          </div>
        </div>
      </div>

      {/* Expandable details section */}
      <div
        onClick={() => setShowDetails(!showDetails)}
        onKeyPress={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setShowDetails(!showDetails);
          }
        }}
        role="button"
        tabIndex={0}
        aria-expanded={showDetails}
        style={{
          padding: '12px 20px',
          background: `${baseColor}0a`,
          borderTop: `1px solid ${baseColor}20`,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          transition: 'background 0.2s ease',
          outline: 'none'
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = `${baseColor}15`}
        onMouseLeave={(e) => e.currentTarget.style.background = `${baseColor}0a`}
        onFocus={(e) => e.currentTarget.style.outline = `3px solid ${baseColor}40`}
        onBlur={(e) => e.currentTarget.style.outline = 'none'}
      >
        <span style={{
          fontSize: '0.9em',
          fontWeight: 500,
          color: '#555'
        }}>
          {showDetails ? 'Hide' : 'Show'} details
        </span>
        {showDetails ? 
          <ChevronUp size={18} color="#555" /> : 
          <ChevronDown size={18} color="#555" />
        }
      </div>

      {/* Details content */}
      {showDetails && (
        <div
          style={{
            padding: '16px 20px',
            background: 'white',
            borderTop: `1px solid ${baseColor}20`,
            animation: 'slideDown 0.3s ease'
          }}
        >
          <ul style={{
            margin: 0,
            padding: '0 0 0 20px',
            fontSize: '0.92em',
            color: '#555',
            lineHeight: '1.8'
          }}>
            {details.map((detail, index) => (
              <li key={index} style={{ marginBottom: '6px' }}>
                {detail}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

function App() {
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState(null);
  const [subtitleFile, setSubtitleFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [taskResult, setTaskResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [files, setFiles] = useState({ upload_files: [], output_files: [] });
  const [showSidebar, setShowSidebar] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [originalSubtitleContent, setOriginalSubtitleContent] = useState('');
  const [syncedSubtitleContent, setSyncedSubtitleContent] = useState('');
  const [activeTab, setActiveTab] = useState('sync'); // 'sync' or 'generate'
  const [syncChecked, setSyncChecked] = useState(true); // Checkbox for sync
  const [overlapChecked, setOverlapChecked] = useState(false); // Checkbox for overlapping issues

  // Repositioning state
  const [isRepositioning, setIsRepositioning] = useState(false);
  const [repositionResult, setRepositionResult] = useState(null);
  const [repositionError, setRepositionError] = useState('');
  const [isRepositioningAnalyzing, setIsRepositioningAnalyzing] = useState(false);
  const [repositioningNeeded, setRepositioningNeeded] = useState(null);
  const [repositioningOutput, setRepositioningOutput] = useState('');

  const [showSettingsPanel, setShowSettingsPanel] = useState(false);
  const [translationEngine, setTranslationEngine] = useState('m2m100');
  const [translationCredentials, setTranslationCredentials] = useState({});

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      const extension = file.name.toLowerCase().split('.').pop();
      
      if (['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'].includes(extension)) {
        setVideoFile(file);
        showMessage(`Video file selected: ${file.name}`, 'success');
      } else if (['srt', 'vtt'].includes(extension)) {
        setSubtitleFile(file);
        showMessage(`Subtitle file selected: ${file.name}`, 'success');
      } else {
        showMessage(`Unsupported file type: ${file.name}`, 'error');
      }
    });
  }, []);

  // Configure react-dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
      'text/*': ['.srt', '.vtt']
    },
    multiple: true
  });

  // Upload files to server
  const uploadFiles = async () => {
    if (!videoFile || !subtitleFile) {
      showMessage('Please select both video and subtitle files', 'error');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Read original subtitle content
      const originalContent = await readSubtitleFile(subtitleFile);
      setOriginalSubtitleContent(originalContent);

      const result = await api.uploadFiles(
        videoFile,
        subtitleFile,
        (progress) => setUploadProgress(progress)
      );
      
      showMessage('Files uploaded successfully!', 'success');
      loadFileList();
    } catch (error) {
      showMessage(`Upload failed: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Handle subtitle generation completion
  const handleSubtitleGenerated = async (outputFile, result) => {
    try {
      // Load the generated subtitle content for preview
      const content = await loadSubtitleContent(outputFile);
      setOriginalSubtitleContent(content);
      
      // Refresh file list to show the new file
      await loadFileList();
      
      // Switch to sync tab for further processing if needed
      showMessage(`Subtitle generated successfully! Switch to "Sync Subtitles" tab to fine-tune timing.`, 'success');
    } catch (error) {
      console.error('Error handling generated subtitle:', error);
    }
  };

  // Handle subtitle updates from preview
  const handleSubtitleUpdate = (updatedSubtitles) => {
    // Convert the updated subtitles array back to SRT format
    const srtContent = updatedSubtitles.map(sub => 
      `${sub.index}\n${sub.startTime} --> ${sub.endTime}\n${sub.text}`
    ).join('\n\n');
    
    setSyncedSubtitleContent(srtContent);
    console.log('Updated subtitles:', srtContent);
    // Here you could save to backend, download file, etc.
  };

  // Load subtitle content from server
  const loadSubtitleContent = async (filename) => {
    try {
      const response = await fetch(`http://localhost:8000/download/${filename}`);
      const content = await response.text();
      return content;
    } catch (error) {
      console.error('Error loading subtitle content:', error);
      return '';
    }
  };

  // Read subtitle file content
  const readSubtitleFile = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  // Start subtitle synchronization process
  const startSynchronization = async () => {
    if (!videoFile || !subtitleFile) {
      showMessage('Please upload files first', 'error');
      return null;
    }

    setIsProcessing(true);
    setTaskStatus(null);
    setTaskResult(null);

    try {
      // Start async sync task
      const startResponse = await api.correctSubtitlesAsync(
        videoFile, 
        subtitleFile,
        translationEngine,
        translationCredentials
      );
      const taskId = startResponse.task_id;
      
      showMessage('Synchronization started!', 'info');

      // Poll for status updates
      await api.waitForSyncCompletion(
        taskId,
        (status) => {
          setTaskStatus({
            status: status.status,
            message: status.current_step || status.message,
            progress: status.progress
          });
        },
        1000 // Poll every 1 second for more responsive updates
      );

      // Get final result
      const finalResult = await api.getSyncResult(taskId);
      setTaskResult({
        success: finalResult.success,
        message: finalResult.message,
        output_file: finalResult.output_file,
        processing_time: finalResult.processing_time
      });

      if (finalResult.success) {
        showMessage('Synchronization completed successfully!', 'success');
        if (finalResult.output_file) {
          const syncedContent = await loadSubtitleContent(finalResult.output_file);
          setSyncedSubtitleContent(syncedContent);
          setShowPreview(true);
        }
        loadFileList();
      } else {
        showMessage(`Synchronization failed: ${finalResult.message}`, 'error');
      }
      return finalResult;
    } catch (error) {
      showMessage(`Synchronization failed: ${error.message}`, 'error');
      return null;
    } finally {
      setIsProcessing(false);
      setTaskStatus(null);
    }
  };

  // Handler to trigger repositioning
  const handleReposition = async (subtitleToUse) => {
    if (!videoFile || !subtitleToUse) {
      showMessage('Please upload both video and subtitle files', 'error');
      return;
    }
    setIsRepositioning(true);
    setRepositionResult(null);
    setRepositionError('');
    try {
      // Use synced subtitle if available, else use original subtitle file
      // let subtitleToUse = (taskResult && taskResult.output_file) ? taskResult.output_file : subtitleFile.name;
      const response = await api.repositionSubtitles({
        video_filename: videoFile.name,
        subtitle_filename: subtitleToUse,
      });
      setRepositionResult(response.output_filename);
      showMessage('Subtitle repositioning completed!', 'success');
      loadFileList();
    } catch (error) {
      setRepositionError(error.response?.data?.detail || error.message);
      showMessage(`Repositioning failed: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsRepositioning(false);
    }
  };

  // Download file from server
  const downloadFile = async (filename) => {
    try {
      await api.downloadFile(filename);
      showMessage(`Downloaded: ${filename}`, 'success');
    } catch (error) {
      showMessage(`Download failed: ${error.message}`, 'error');
    }
  };

  // Function to show messages
  const showMessage = (msg, type) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 5000);
  };

  // Get status icon based on status
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />;
    }
  };

  // Load file list from server
  const loadFileList = async () => {
    try {
      const fileList = await api.listFiles();
      setFiles(fileList);
    } catch (error) {
      console.error('Failed to load file list:', error);
    }
  };

  // Load file list on component mount
  React.useEffect(() => {
    loadFileList();
  }, []);

  // Delete file from server
  const deleteFile = async (filename, fileType) => {
    try {
      await api.deleteFile(filename, fileType);
      showMessage(`Deleted: ${filename}`, 'success');
      loadFileList();
    } catch (error) {
      showMessage(`Delete failed: ${error.message}`, 'error');
    }
  };

  // Clear selected files and reset states
  const clearFiles = () => {
    setVideoFile(null);
    setSubtitleFile(null);
    setTaskId(null);
    setTaskStatus(null);
    setTaskResult(null);
    // reposition states:
    setRepositionResult(null);
    setRepositionError('');
    setSyncedSubtitleContent('');
    setOriginalSubtitleContent('');
  };

  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/preview" element={<SubtitlePreviewPage />} />
    </Routes>
  );
}

export default App;

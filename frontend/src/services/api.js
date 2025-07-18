import axios from 'axios';

const API_URL = 'http://localhost:8000';

/**
 * Sube un archivo nuevo o reutilizado.
 * Si se pasa un `FormData`, lo usa directamente.
 * Si se pasa un `File`, lo empaqueta en un `FormData`.
 */

// Add these new endpoints

export const getBoxPlot = async (trait) => {
  const response = await axios.get(`${API_URL}/plot/boxplot?trait=${trait}`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};


export const uploadDataset = async (fileOrFormData, isFormData = false) => {
  const formData = isFormData ? fileOrFormData : new FormData();
  if (!isFormData) formData.append('file', fileOrFormData);

  const response = await axios.post(`${API_URL}/upload/`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const analyzeData = async (config) => {
  const response = await axios.post(`${API_URL}/analyze/`, config);
  return response.data;
};

export const getResults = async () => {
  const response = await axios.get(`${API_URL}/results/`);
  return response.data;
};

export const exportToExcel = async () => {
  const response = await axios.get(`${API_URL}/export/excel`, {
    responseType: 'blob',
  });
  return response.data;
};

export const exportToPDF = async () => {
  const response = await axios.get(`${API_URL}/export/pdf`, {
    responseType: 'blob',
  });
  return response.data;
};

export const getClusterDistributionPlot = async () => {
  const response = await axios.get(`${API_URL}/plot/cluster_distribution`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};

export const getTraitDistributionPlot = async (trait) => {
  const response = await axios.get(`${API_URL}/plot/trait_distribution?trait=${trait}`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};

export const getPCAPlot = async () => {
  const response = await axios.get(`${API_URL}/plot/pca`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};

// New function for radar plot
export const getRadarPlot = async () => {
  const response = await axios.get(`${API_URL}/plot/radar`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};

// New function for loading sample data
export const loadSampleData = async () => {
  const response = await axios.get(`${API_URL}/load-sample-data`);
  return response.data;
};
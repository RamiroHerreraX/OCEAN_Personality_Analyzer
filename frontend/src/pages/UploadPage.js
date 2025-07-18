// src/pages/UploadPage.js
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../context/AnalysisContext';
import { uploadDataset } from '../services/api';

import {
  Button,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
  Stack,
} from '@mui/material';

import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ReplayIcon from '@mui/icons-material/Replay';
import DeleteIcon from '@mui/icons-material/Delete';

const UploadPage = ({ setLoading }) => {
  const { setDataset } = useAnalysis();
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [uploadHistory, setUploadHistory] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const savedHistory = JSON.parse(localStorage.getItem('uploadHistory')) || [];
    setUploadHistory(savedHistory);
  }, []);

  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const saveToHistory = async (file) => {
    const base64 = await toBase64(file);
    const newEntry = {
      name: file.name,
      timestamp: new Date().toISOString(),
      content: base64,
      type: file.type,
    };

    const updatedHistory = [newEntry, ...uploadHistory.slice(0, 9)];
    setUploadHistory(updatedHistory);
    localStorage.setItem('uploadHistory', JSON.stringify(updatedHistory));
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);

      const result = await uploadDataset(formData, true);
      setDataset(result.preview);
      setStats(result.stats);

      await saveToHistory(file);
      setFile(null);

      setLoading(false);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Error al cargar el archivo');
      setLoading(false);
    }
  };

  const handleReuse = async (item) => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(item.content);
      const blob = await response.blob();
      const reusedFile = new File([blob], item.name, { type: item.type });

      const formData = new FormData();
      formData.append('file', reusedFile);

      const result = await uploadDataset(formData, true);
      setDataset(result.preview);
      setStats(result.stats);

      setLoading(false);
    } catch (err) {
      console.error(err);
      setError('Error al reutilizar el archivo');
      setLoading(false);
    }
  };

  const handleDelete = (index) => {
    const updatedHistory = [...uploadHistory];
    updatedHistory.splice(index, 1);
    setUploadHistory(updatedHistory);
    localStorage.setItem('uploadHistory', JSON.stringify(updatedHistory));
  };

  const handleContinue = () => {
    navigate('/analyze');
  };

  return (
    <Box maxWidth="lg" mx="auto" px={2} py={4}>
      <Typography variant="h4" gutterBottom>
        📁 Cargar Dataset
      </Typography>

      <Typography variant="body1" paragraph>
        Sube un archivo <strong>CSV</strong> o <strong>Excel</strong> con datos de personalidad OCEAN para comenzar el análisis.
      </Typography>

      {error && (
        <Paper elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#ffebee' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      {/* Historial */}
      {uploadHistory.length > 0 && (
        <Box mb={4}>
          <Typography variant="h6">📜 Historial de Archivos Subidos</Typography>
          <TableContainer component={Paper} sx={{ mt: 2 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Nombre del Archivo</TableCell>
                  <TableCell>Fecha y Hora</TableCell>
                  <TableCell>Acciones</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {uploadHistory.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{item.name}</TableCell>
                    <TableCell>{new Date(item.timestamp).toLocaleString()}</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1}>
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<ReplayIcon />}
                          onClick={() => handleReuse(item)}
                        >
                          Usar
                        </Button>
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          startIcon={<DeleteIcon />}
                          onClick={() => handleDelete(index)}
                        >
                          Eliminar
                        </Button>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      <Grid container spacing={4}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Selección y Carga del Archivo
              </Typography>

              <Box display="flex" alignItems="center" flexWrap="wrap" gap={2} mb={3}>
                <input
                  accept=".csv,.xlsx,.xls"
                  style={{ display: 'none' }}
                  id="upload-file"
                  type="file"
                  onChange={handleFileChange}
                />
                <label htmlFor="upload-file">
                  <Button
                    variant="contained"
                    color="primary"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                  >
                    Seleccionar Archivo
                  </Button>
                </label>

                {file && (
                  <Box display="flex" alignItems="center" gap={1}>
                    <DescriptionIcon color="action" />
                    <Typography variant="body1">{file.name}</Typography>
                  </Box>
                )}
              </Box>

              <Button
                variant="contained"
                color="secondary"
                onClick={handleUpload}
                disabled={!file}
                startIcon={<CloudUploadIcon />}
              >
                Subir y Procesar
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {stats && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <CheckCircleIcon color="primary" />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Dataset cargado exitosamente
                  </Typography>
                </Box>

                <Typography variant="subtitle1" gutterBottom>
                  Resumen Estadístico:
                </Typography>

                <TableContainer component={Paper} sx={{ mb: 2 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Estadística</TableCell>
                        {Object.keys(stats.summary).map((col) => (
                          <TableCell key={col} align="right">
                            {col}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(stats.summary['EXT1']).map(([stat]) => (
                        <TableRow key={stat}>
                          <TableCell component="th" scope="row">
                            {stat}
                          </TableCell>
                          {Object.keys(stats.summary).map((col) => (
                            <TableCell key={`${stat}-${col}`} align="right">
                              {stats.summary[col][stat].toFixed(2)}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Button variant="contained" color="primary" onClick={handleContinue}>
                  Continuar al Análisis
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default UploadPage;

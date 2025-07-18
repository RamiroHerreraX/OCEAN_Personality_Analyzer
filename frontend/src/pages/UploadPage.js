import React, { useState } from 'react';
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
} from '@mui/material';

import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const UploadPage = ({ setLoading }) => {
  const { dataset, setDataset } = useAnalysis();
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    try {
      setLoading(true);
      setError(null);

      const result = await uploadDataset(file);
      setDataset(result.preview);
      setStats(result.stats);

      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al cargar el archivo');
      setLoading(false);
    }
  };

  const handleContinue = () => {
    navigate('/analyze');
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Cargar Dataset
      </Typography>

      <Typography variant="body1" paragraph>
        Sube un archivo CSV o Excel con los datos de personalidad OCEAN para comenzar el análisis.
      </Typography>

      {error && (
        <Paper
          elevation={2}
          sx={{
            p: 2,
            mb: 2,
            backgroundColor: '#ffebee',
          }}
        >
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
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
              <Box ml={2} display="flex" alignItems="center">
                <DescriptionIcon color="action" />
                <Typography variant="body1" sx={{ ml: 1 }}>
                  {file.name}
                </Typography>
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

      {stats && (
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

            <Button variant="contained" color="primary" onClick={handleContinue} sx={{ mt: 2 }}>
              Continuar al Análisis
            </Button>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default UploadPage;

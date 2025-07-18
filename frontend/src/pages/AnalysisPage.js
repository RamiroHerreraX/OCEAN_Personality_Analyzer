import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../context/AnalysisContext';
import { analyzeData } from '../services/api';

import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  ListItemText,
  Paper,
  Grid,
  FormControlLabel,
  Slider,
} from '@mui/material';

import PlayCircleFilledIcon from '@mui/icons-material/PlayCircleFilled';
import Alert from '@mui/material/Alert';

const AnalysisPage = ({ setLoading }) => {
  const navigate = useNavigate();
  const { dataset, config, setConfig, setResults } = useAnalysis();
  const [error, setError] = useState(null);
  const [availableFeatures, setAvailableFeatures] = useState([]);

  useEffect(() => {
    if (!dataset) {
      navigate('/');
      return;
    }

    // Extraer características disponibles (columnas que comienzan con EXT, EST, etc.)
    const traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN'];
    const features = [];

    traits.forEach((trait) => {
      for (let i = 1; i <= 10; i++) {
        const col = `${trait}${i}`;
        if (dataset[0] && dataset[0][col] !== undefined) {
          features.push(col);
        }
      }
    });

    setAvailableFeatures(features);
    setConfig((prev) => ({
      ...prev,
      features: features.length > 0 ? features.slice(0, 10) : [],
    }));
  }, [dataset, navigate, setConfig]);

  const handleFeatureChange = (event) => {
    setConfig((prev) => ({
      ...prev,
      features: event.target.value,
    }));
  };

  const handleClusterChange = (event, newValue) => {
    setConfig((prev) => ({
      ...prev,
      clusters: newValue,
    }));
  };

  const handleScaleChange = (event) => {
    setConfig((prev) => ({
      ...prev,
      scale: event.target.checked,
    }));
  };

  const handlePCASliderChange = (event, newValue) => {
    setConfig((prev) => ({
      ...prev,
      pcaComponents: newValue === 0 ? null : newValue,
    }));
  };

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);

      const result = await analyzeData(config);
      setResults(result);

      setLoading(false);
      navigate('/results');
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al realizar el análisis');
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Configurar Análisis
      </Typography>

      <Typography variant="body1" paragraph>
        Configura los parámetros para el análisis de clusters usando el algoritmo K-Means.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Parámetros del Modelo
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="features-label">Variables a Incluir</InputLabel>
                <Select
                  labelId="features-label"
                  id="features-select"
                  multiple
                  value={config.features}
                  onChange={handleFeatureChange}
                  renderValue={(selected) => selected.join(', ')}
                >
                  {availableFeatures.map((feature) => (
                    <MenuItem key={feature} value={feature}>
                      <Checkbox checked={config.features.indexOf(feature) > -1} />
                      <ListItemText primary={feature} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography id="cluster-slider" gutterBottom>
                Número de Clusters: {config.clusters}
              </Typography>
              <Slider
                value={config.clusters}
                onChange={handleClusterChange}
                aria-labelledby="cluster-slider"
                step={1}
                marks
                min={2}
                max={10}
                valueLabelDisplay="auto"
                sx={{ width: '100%', mt: 2 }}
              />

              <FormControlLabel
                control={
                  <Checkbox checked={config.scale} onChange={handleScaleChange} color="primary" />
                }
                label="Estandarizar variables (MinMax Scaling)"
              />

              <Typography id="pca-slider" gutterBottom sx={{ mt: 2 }}>
                Componentes PCA: {config.pcaComponents || 'Ninguno'}
              </Typography>
              <Slider
                value={config.pcaComponents || 0}
                onChange={handlePCASliderChange}
                aria-labelledby="pca-slider"
                step={1}
                marks={[
                  { value: 0, label: 'Ninguno' },
                  { value: 2, label: '2' },
                  { value: 3, label: '3' },
                ]}
                min={0}
                max={3}
                valueLabelDisplay="auto"
                sx={{ width: '100%' }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Vista Previa de Datos
              </Typography>

              {dataset && dataset.length > 0 ? (
                <Paper sx={{ maxHeight: 400, overflow: 'auto', p: 2 }}>
                  <pre style={{ fontSize: '0.8rem' }}>{JSON.stringify(dataset.slice(0, 5), null, 2)}</pre>
                </Paper>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No hay datos disponibles
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box display="flex" justifyContent="flex-end" mt={2}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<PlayCircleFilledIcon />}
          onClick={handleAnalyze}
          disabled={config.features.length === 0}
        >
          Ejecutar Análisis
        </Button>
      </Box>
    </Box>
  );
};

export default AnalysisPage;

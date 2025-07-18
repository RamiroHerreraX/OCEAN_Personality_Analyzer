import React, { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import { analyzeData } from "../services/api";

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
  Divider,
  Alert,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";

import PlayCircleFilledIcon from "@mui/icons-material/PlayCircleFilled";

const AnalysisPage = ({ setLoading }) => {
  const navigate = useNavigate();
  const { dataset, config, setConfig, setResults } = useAnalysis();
  const [error, setError] = useState(null);
  const [availableFeatures, setAvailableFeatures] = useState([]);

  useEffect(() => {
    if (!dataset) {
      navigate("/");
      return;
    }

    const traits = ["EXT", "EST", "AGR", "CSN", "OPN"];
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

  const groupedFeatures = useMemo(() => {
    const groups = { EXT: [], EST: [], AGR: [], CSN: [], OPN: [] };
    availableFeatures.forEach((feature) => {
      const group = feature.slice(0, 3);
      if (groups[group]) {
        groups[group].push(feature);
      }
    });
    return groups;
  }, [availableFeatures]);

  const handleFeatureChange = (event) => {
    setConfig((prev) => ({
      ...prev,
      features: event.target.value,
    }));
  };

  const handleToggleGroup = (groupKey) => {
    const groupFeatures = groupedFeatures[groupKey];
    const allSelected = groupFeatures.every((f) =>
      config.features.includes(f)
    );

    setConfig((prev) => ({
      ...prev,
      features: allSelected
        ? prev.features.filter((f) => !groupFeatures.includes(f))
        : [...new Set([...prev.features, ...groupFeatures])],
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
      navigate("/results");
    } catch (err) {
      setError(err.response?.data?.detail || "Error al realizar el análisis");
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, backgroundColor: "#f7f9fc", minHeight: "100vh" }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{ fontWeight: "bold", mb: 2, color: "#2c3e50" }}
      >
        Configurar Análisis
      </Typography>

      <Typography variant="body1" paragraph sx={{ color: "#34495e", mb: 3 }}>
        Configura los parámetros para el análisis de clusters usando el
        algoritmo K-Means.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuración */}
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3, borderRadius: 3, boxShadow: 3 }}>
            <CardContent>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ color: "#2980b9", fontWeight: "bold" }}
              >
                Parámetros del Modelo
              </Typography>

              <Divider sx={{ my: 2 }} />

              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel id="features-label">Variables a Incluir</InputLabel>
                <Select
                  labelId="features-label"
                  id="features-select"
                  multiple
                  value={config.features}
                  onChange={handleFeatureChange}
                  renderValue={(selected) => (
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Stack>
                  )}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 300,
                      },
                    },
                  }}
                >
                  {Object.entries(groupedFeatures).map(([groupKey, features]) => {
                    const allSelected = features.every((f) =>
                      config.features.includes(f)
                    );
                    const someSelected = features.some((f) =>
                      config.features.includes(f)
                    );

                    return (
                      <Box key={groupKey}>
                        <MenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleGroup(groupKey);
                          }}
                        >
                          <Checkbox
                            checked={allSelected}
                            indeterminate={!allSelected && someSelected}
                          />
                          <ListItemText
                            primary={`Grupo ${groupKey}`}
                            primaryTypographyProps={{ fontWeight: "bold" }}
                          />
                        </MenuItem>
                        {features.map((feature) => (
                          <MenuItem key={feature} value={feature}>
                            <Checkbox checked={config.features.includes(feature)} />
                            <ListItemText primary={feature} />
                          </MenuItem>
                        ))}
                      </Box>
                    );
                  })}
                </Select>
                <Typography variant="caption" color="text.secondary">
                  Selecciona variables individuales o grupos completos para el análisis
                </Typography>
              </FormControl>

              <Box sx={{ mb: 3 }}>
                <Typography id="cluster-slider" gutterBottom>
                  Número de Clusters: <strong>{config.clusters}</strong>
                </Typography>
                <Slider
                  value={config.clusters}
                  onChange={handleClusterChange}
                  aria-labelledby="cluster-slider"
                  step={1}
                  marks
                  min={1}
                  max={5}
                  valueLabelDisplay="auto"
                  sx={{ width: "95%", mt: 2, ml: 1 }}
                />
              </Box>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={config.scale}
                    onChange={handleScaleChange}
                    color="primary"
                  />
                }
                label={
                  <Typography variant="body2">
                    Estandarizar variables (MinMax Scaling)
                  </Typography>
                }
                sx={{ mt: 1, mb: 2 }}
              />

              <Box sx={{ mt: 3 }}>
                <Typography id="pca-slider" gutterBottom>
                  Componentes PCA:{" "}
                  <strong>{config.pcaComponents || "Ninguno"}</strong>
                </Typography>
                <Slider
                  value={config.pcaComponents || 0}
                  onChange={handlePCASliderChange}
                  aria-labelledby="pca-slider"
                  step={1}
                  marks={[
                    { value: 0, label: "Ninguno" },
                    { value: 2, label: "2" },
                    { value: 3, label: "3" },
                  ]}
                  min={0}
                  max={3}
                  valueLabelDisplay="auto"
                  sx={{ width: "95%", ml: 1 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Vista previa */}
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3, borderRadius: 3, boxShadow: 3 }}>
            <CardContent>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ color: "#2980b9", fontWeight: "bold" }}
              >
                Vista Previa de Datos
              </Typography>

              <Divider sx={{ my: 2 }} />

              {dataset && dataset.length > 0 ? (
                <Paper
                  sx={{
                    maxHeight: 500,
                    overflow: "auto",
                    p: 2,
                    backgroundColor: "#f8fafc",
                    borderRadius: 2,
                    border: "1px solid #e0e0e0",
                  }}
                >
                  <TableContainer>
                    <Table size="small" aria-label="data preview">
                      <TableHead>
                        <TableRow sx={{ backgroundColor: "#e3f2fd" }}>
                          {Object.keys(dataset[0])
                            .slice(0, 6)
                            .map((key) => (
                              <TableCell key={key} sx={{ fontWeight: "bold" }}>
                                {key}
                              </TableCell>
                            ))}
                          {Object.keys(dataset[0]).length > 6 && (
                            <TableCell sx={{ fontWeight: "bold" }}>...</TableCell>
                          )}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {dataset.slice(0, 5).map((row, index) => (
                          <TableRow key={index}>
                            {Object.values(row)
                              .slice(0, 6)
                              .map((value, i) => (
                                <TableCell key={i}>
                                  {typeof value === "object"
                                    ? JSON.stringify(value)
                                    : value}
                                </TableCell>
                              ))}
                            {Object.keys(row).length > 6 && <TableCell>...</TableCell>}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 1, display: "block" }}
                  >
                    Mostrando 5 filas y{" "}
                    {Math.min(6, Object.keys(dataset[0] || {}).length)} columnas
                    de {dataset.length} filas totales
                  </Typography>
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

      {/* Botón + resumen */}
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mt={4}
        flexWrap="wrap"
      >
        <Box
          sx={{
            backgroundColor: "#e3f2fd",
            p: 2,
            borderRadius: 2,
            flexGrow: 1,
            mr: { md: 3 },
            mb: { xs: 2, md: 0 },
          }}
        >
          <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
            <strong>Resumen de configuración:</strong>
          </Typography>
          <Stack direction="row" spacing={1} mt={1} flexWrap="wrap">
            <Chip
              label={`${config.features.length} variables`}
              color="primary"
              variant="outlined"
              size="small"
            />
            <Chip
              label={`${config.clusters} clusters`}
              color="primary"
              variant="outlined"
              size="small"
            />
            <Chip
              label={`PCA: ${config.pcaComponents || "Ninguno"}`}
              color="primary"
              variant="outlined"
              size="small"
            />
            <Chip
              label={config.scale ? "Escalado: Sí" : "Escalado: No"}
              color="primary"
              variant="outlined"
              size="small"
            />
          </Stack>
        </Box>

        <Button
          variant="contained"
          size="large"
          sx={{
            backgroundColor: "#1976d2",
            borderRadius: 3,
            px: 4,
            py: 1.5,
            textTransform: "none",
            fontSize: "1rem",
            fontWeight: "medium",
            "&:hover": {
              backgroundColor: "#1565c0",
              boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
            },
          }}
          startIcon={<PlayCircleFilledIcon sx={{ fontSize: "1.5rem" }} />}
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

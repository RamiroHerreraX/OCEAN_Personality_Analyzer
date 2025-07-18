import React, { useState, useEffect } from 'react';
import { useAnalysis } from '../context/AnalysisContext';
import {
  getResults,
  exportToExcel,
  exportToPDF,
  getClusterDistributionPlot,
  getTraitDistributionPlot,
  getPCAPlot,
} from '../services/api';

import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
} from '@mui/material';

import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import TableChartIcon from '@mui/icons-material/TableChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import SaveAltIcon from '@mui/icons-material/SaveAlt';

function TabPanel({ children, value, index }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

const ResultsPage = ({ setLoading }) => {
  const { results, setResults } = useAnalysis();
  const [tabValue, setTabValue] = useState(0);
  const [clusterPlot, setClusterPlot] = useState(null);
  const [extPlot, setExtPlot] = useState(null);
  const [estPlot, setEstPlot] = useState(null);
  const [agrPlot, setAgrPlot] = useState(null);
  const [csnPlot, setCsnPlot] = useState(null);
  const [opnPlot, setOpnPlot] = useState(null);
  const [pcaPlot, setPcaPlot] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!results) {
      const fetchResults = async () => {
        try {
          setLoading(true);
          const data = await getResults();
          setResults(data);
          setLoading(false);
        } catch (err) {
          console.error(err);
          setError('Error al cargar los resultados');
          setLoading(false);
        }
      };
      fetchResults();
    }
  }, [results, setResults, setLoading]);

  useEffect(() => {
    if (results) {
      const loadPlots = async () => {
        setLoading(true);
        try {
          setClusterPlot(await getClusterDistributionPlot());
          setExtPlot(await getTraitDistributionPlot('EXT'));
          setEstPlot(await getTraitDistributionPlot('EST'));
          setAgrPlot(await getTraitDistributionPlot('AGR'));
          setCsnPlot(await getTraitDistributionPlot('CSN'));
          setOpnPlot(await getTraitDistributionPlot('OPN'));
          setPcaPlot(await getPCAPlot());
        } catch (err) {
          console.error(err);
          setError('Error al cargar las visualizaciones');
        }
        setLoading(false);
      };
      loadPlots();
    }
  }, [results, setLoading]);

  const handleExport = async (type) => {
    try {
      setLoading(true);
      const blob = type === 'excel' ? await exportToExcel() : await exportToPDF();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = type === 'excel' ? 'ocean_results.xlsx' : 'ocean_results.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError(`Error al exportar a ${type.toUpperCase()}`);
    } finally {
      setLoading(false);
    }
  };

  if (!results) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', minHeight: 300 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', minHeight: 300 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Resultados del Análisis</Typography>
        <Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={<SaveAltIcon />}
            onClick={() => handleExport('excel')}
            sx={{ mr: 2 }}
          >
            Exportar Excel
          </Button>
          <Button
            variant="contained"
            color="secondary"
            startIcon={<PictureAsPdfIcon />}
            onClick={() => handleExport('pdf')}
          >
            Exportar PDF
          </Button>
        </Box>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Resumen del Análisis
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography>
                <strong>Configuración:</strong> {results?.config?.clusters || 'N/A'} clusters,{' '}
                {results?.config?.features?.length || 0} variables
              </Typography>

              <TableContainer component={Paper} sx={{ maxHeight: 400, my: 2 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Cluster</TableCell>
                      <TableCell align="right">Cantidad</TableCell>
                      <TableCell align="right">Porcentaje</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {results?.cluster_counts &&
                      Object.entries(results.cluster_counts).map(([cluster, count]) => (
                        <TableRow key={cluster}>
                          <TableCell>{cluster}</TableCell>
                          <TableCell align="right">{count}</TableCell>
                          <TableCell align="right">
                            {((count / (results?.sample_data?.length || 1)) * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>

            <Grid item xs={12} md={6}>
              {clusterPlot && (
                <Box
                  sx={{
                    height: 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: '#f5f5f5',
                    borderRadius: 1,
                    p: 2,
                  }}
                >
                  <img
                    src={clusterPlot}
                    alt="Distribución de Clusters"
                    style={{ maxWidth: '100%', maxHeight: '100%' }}
                  />
                </Box>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Tabs
        value={tabValue}
        onChange={(_, newValue) => setTabValue(newValue)}
        variant="scrollable"
        scrollButtons="auto"
        textColor="primary"
        indicatorColor="primary"
      >
        <Tab icon={<BarChartIcon />} label="Extroversión" />
        <Tab icon={<PieChartIcon />} label="Neuroticismo" />
        <Tab icon={<BarChartIcon />} label="Amabilidad" />
        <Tab icon={<PieChartIcon />} label="Responsabilidad" />
        <Tab icon={<BarChartIcon />} label="Apertura" />
        <Tab icon={<ScatterPlotIcon />} label="PCA" />
        <Tab icon={<TableChartIcon />} label="Datos" />
      </Tabs>

      {[
        { index: 0, title: 'Distribución de Extroversión', img: extPlot },
        { index: 1, title: 'Distribución de Neuroticismo', img: estPlot },
        { index: 2, title: 'Distribución de Amabilidad', img: agrPlot },
        { index: 3, title: 'Distribución de Responsabilidad', img: csnPlot },
        { index: 4, title: 'Distribución de Apertura', img: opnPlot },
        { index: 5, title: 'Visualización PCA', img: pcaPlot },
      ].map(({ index, title, img }) => (
        <TabPanel key={index} value={tabValue} index={index}>
          <Typography variant="h6" gutterBottom>{title}</Typography>
          {img ? (
            <Box
              sx={{
                height: 400,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                bgcolor: '#f5f5f5',
                borderRadius: 1,
                p: 2,
              }}
            >
              <img src={img} alt={title} style={{ maxHeight: '100%', maxWidth: '100%' }} />
            </Box>
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', minHeight: 300 }}>
              <CircularProgress />
            </Box>
          )}
        </TabPanel>
      ))}

      <TabPanel value={tabValue} index={6}>
        <Typography variant="h6" gutterBottom>Datos de Muestra (primeras 50 filas)</Typography>
        {results?.sample_data ? (
          <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  {results.sample_data.length > 0 &&
                    Object.keys(results.sample_data[0]).map((key) => (
                      <TableCell key={key}>{key}</TableCell>
                    ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {results.sample_data.map((row, i) => (
                  <TableRow key={i}>
                    {Object.values(row).map((val, j) => (
                      <TableCell key={j}>
                        {typeof val === 'object' ? JSON.stringify(val) : val}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', minHeight: 300 }}>
            <CircularProgress />
          </Box>
        )}
      </TabPanel>
    </Box>
  );
};

export default ResultsPage;

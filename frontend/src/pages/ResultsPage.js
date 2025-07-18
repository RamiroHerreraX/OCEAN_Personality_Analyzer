import React, { useState, useEffect } from 'react';
import { useAnalysis } from '../context/AnalysisContext';
import {
  getResults,
  exportToExcel,
  exportToPDF,
  getClusterDistributionPlot,
  getTraitDistributionPlot,
  getPCAPlot,
  getRadarPlot,
  getBoxPlot
} from '../services/api';
import { getTraitName } from '../utils/traits';
import PersonalityTypeCard from '../components/PersonalityTypeCard';

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

// Importaciones de iconos
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import TableChartIcon from '@mui/icons-material/TableChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import RadarIcon from '@mui/icons-material/Radar';
import ShowChartIcon from '@mui/icons-material/ShowChart'; // Usaremos este para box plots
import PsychologyIcon from '@mui/icons-material/Psychology';

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
  const { results, setResults, personalityTypes, setPersonalityTypes } = useAnalysis();
  const [tabValue, setTabValue] = useState(0);
  const [clusterPlot, setClusterPlot] = useState(null);
  const [extPlot, setExtPlot] = useState(null);
  const [estPlot, setEstPlot] = useState(null);
  const [agrPlot, setAgrPlot] = useState(null);
  const [csnPlot, setCsnPlot] = useState(null);
  const [opnPlot, setOpnPlot] = useState(null);
  const [pcaPlot, setPcaPlot] = useState(null);
  const [radarPlot, setRadarPlot] = useState(null);
  const [boxPlots, setBoxPlots] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    if (results?.personality_types) {
      setPersonalityTypes(results.personality_types);
    }
  }, [results, setPersonalityTypes]);

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
          const [
            clusterPlot, 
            extPlot, 
            estPlot, 
            agrPlot, 
            csnPlot, 
            opnPlot, 
            pcaPlot,
            radarPlot
          ] = await Promise.all([
            getClusterDistributionPlot(),
            getTraitDistributionPlot('EXT'),
            getTraitDistributionPlot('EST'),
            getTraitDistributionPlot('AGR'),
            getTraitDistributionPlot('CSN'),
            getTraitDistributionPlot('OPN'),
            getPCAPlot(),
            getRadarPlot()
          ]);
          
          setClusterPlot(clusterPlot);
          setExtPlot(extPlot);
          setEstPlot(estPlot);
          setAgrPlot(agrPlot);
          setCsnPlot(csnPlot);
          setOpnPlot(opnPlot);
          setPcaPlot(pcaPlot);
          setRadarPlot(radarPlot);

          // Load box plots for each trait
          const boxPlotData = {};
          for (const trait of ['EXT', 'EST', 'AGR', 'CSN', 'OPN']) {
            boxPlotData[trait] = await getBoxPlot(trait);
          }
          setBoxPlots(boxPlotData);
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
        <Tab icon={<RadarIcon />} label="Radar" />
        <Tab icon={<ShowChartIcon />} label="Box Plots" />
        <Tab icon={<TableChartIcon />} label="Datos" />
        <Tab icon={<PsychologyIcon />} label="Tipos Personalidad" />
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
          {index === 5 && (
            <Box mt={2}>
              <Typography variant="body2" color="text.secondary">
                Esta visualización muestra la reducción dimensional de los datos usando PCA. 
                Los colores representan los diferentes clusters y las X marcan los centroides.
              </Typography>
            </Box>
          )}
        </TabPanel>
      ))}

      <TabPanel value={tabValue} index={6}>
        <Typography variant="h6" gutterBottom>Perfil de Personalidad por Cluster</Typography>
        {radarPlot ? (
          <Box sx={{ height: 500, display: 'flex', justifyContent: 'center' }}>
            <img src={radarPlot} alt="Radar Plot" style={{ maxHeight: '100%' }} />
          </Box>
        ) : (
          <CircularProgress />
        )}
        <Box mt={2}>
          <Typography variant="body1">
            Este gráfico radar muestra el perfil promedio de cada cluster en los 5 rasgos OCEAN.
          </Typography>
          <Typography variant="body2" color="text.secondary" mt={1}>
            Los valores más cercanos al borde exterior indican mayor presencia del rasgo.
          </Typography>
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={7}>
        <Typography variant="h6" gutterBottom>Distribución de Rasgos por Cluster</Typography>
        {Object.keys(boxPlots).length > 0 ? (
          <Grid container spacing={2}>
            {['EXT', 'EST', 'AGR', 'CSN', 'OPN'].map((trait) => (
              <Grid item xs={12} md={6} key={trait}>
                <Typography variant="subtitle1" align="center">
                  {getTraitName(trait)}
                </Typography>
                <Box sx={{ height: 300 }}>
                  <img 
                    src={boxPlots[trait]} 
                    alt={`Box Plot ${trait}`} 
                    style={{ maxHeight: '100%', width: '100%' }} 
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        ) : (
          <CircularProgress />
        )}
        <Box mt={2}>
          <Typography variant="body2" color="text.secondary">
            Los diagramas de caja muestran la distribución de cada rasgo por cluster.
            La línea central representa la mediana, la caja el rango intercuartílico y los bigotes el rango total.
          </Typography>
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={8}>
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

      <TabPanel value={tabValue} index={9}>
        <Typography variant="h6" gutterBottom>Tipos de Personalidad por Cluster</Typography>
        <Grid container spacing={2}>
          {Object.entries(personalityTypes).map(([cluster, type]) => (
            <Grid item xs={12} sm={6} md={4} key={cluster}>
              <PersonalityTypeCard cluster={cluster} type={type} />
            </Grid>
          ))}
        </Grid>
        <Box mt={2}>
          <Typography variant="body2" color="text.secondary">
            Cada cluster ha sido asignado a un tipo de personalidad basado en sus rasgos dominantes OCEAN.
          </Typography>
        </Box>
      </TabPanel>
    </Box>
  );
};

export default ResultsPage;
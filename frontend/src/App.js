import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';

import PsychologyIcon from '@mui/icons-material/Psychology';

import UploadPage from './pages/UploadPage';
import AnalysisPage from './pages/AnalysisPage';
import ResultsPage from './pages/ResultsPage';
import { AnalysisProvider } from './context/AnalysisContext';

const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [loading, setLoading] = useState(false);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AnalysisProvider>
        <Router>
          <AppBar position="static">
            <Toolbar>
              <IconButton edge="start" color="inherit" aria-label="menu" sx={{ mr: 2 }}>
                <PsychologyIcon />
              </IconButton>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                OCEAN Personality Analyzer
              </Typography>
            </Toolbar>
            {loading && <LinearProgress color="secondary" />}
          </AppBar>

          <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Routes>
              <Route path="/" element={<UploadPage setLoading={setLoading} />} />
              <Route path="/analyze" element={<AnalysisPage setLoading={setLoading} />} />
              <Route path="/results" element={<ResultsPage setLoading={setLoading} />} />
            </Routes>
          </Container>
        </Router>
      </AnalysisProvider>
    </ThemeProvider>
  );
}

export default App;




/*import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;*/

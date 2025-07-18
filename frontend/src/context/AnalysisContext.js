// src/context/AnalysisContext.js
import React, { createContext, useState, useContext } from 'react';

const AnalysisContext = createContext();

export const AnalysisProvider = ({ children }) => {
  const [dataset, setDataset] = useState(null);
  const [results, setResults] = useState(null);
  const [config, setConfig] = useState({
    clusters: 5,
    features: [],
    scale: true,
    pcaComponents: null
  });
  const [personalityTypes, setPersonalityTypes] = useState({});
  const [history, setHistory] = useState([]);

  return (
    <AnalysisContext.Provider 
      value={{ 
        dataset, 
        setDataset, 
        results, 
        setResults, 
        config, 
        setConfig,
        personalityTypes,
        setPersonalityTypes,
        history,
        setHistory
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => useContext(AnalysisContext);
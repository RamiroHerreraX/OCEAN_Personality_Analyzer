// src/utils/history.js
export const getHistory = () => {
  return JSON.parse(localStorage.getItem('uploadHistory')) || [];
};

export const saveToHistory = (file) => {
  const history = getHistory();
  const newEntry = {
    name: file.name,
    timestamp: new Date().toISOString(),
    size: file.size,
    type: file.type
  };
  
  const updatedHistory = [newEntry, ...history.slice(0, 9)];
  localStorage.setItem('uploadHistory', JSON.stringify(updatedHistory));
  return updatedHistory;
};

export const removeFromHistory = (index) => {
  const history = getHistory();
  history.splice(index, 1);
  localStorage.setItem('uploadHistory', JSON.stringify(history));
  return history;
};
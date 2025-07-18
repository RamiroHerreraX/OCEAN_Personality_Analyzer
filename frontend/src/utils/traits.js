// src/utils/traits.js
export const getTraitName = (code) => {
  const traits = {
    'EXT': 'Extraversión',
    'EST': 'Neuroticismo',
    'AGR': 'Amabilidad',
    'CSN': 'Responsabilidad',
    'OPN': 'Apertura'
  };
  return traits[code] || code;
};

export const getTraitDescription = (code) => {
  const descriptions = {
    'EXT': 'Grado de sociabilidad, hablador y energía',
    'EST': 'Grado de estabilidad emocional y sensibilidad',
    'AGR': 'Grado de cooperación y compasión',
    'CSN': 'Grado de organización y autodisciplina',
    'OPN': 'Grado de creatividad y curiosidad'
  };
  return descriptions[code] || '';
};